# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement a multiprocess PPOCritic
"""

import logging
import os

import torch
import torch.distributed
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tensordict import TensorDict
from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import masked_mean
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.critic import BasePPOCritic

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOCritic(BasePPOCritic):
    def __init__(self, config, critic_module: nn.Module, critic_optimizer: optim.Optimizer):
        super().__init__(config=config)
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        print(f"Critic use_remove_padding={self.use_remove_padding}")

        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        self.device_name = get_device_name()
        self.bce_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def _forward_prompt_value(self, micro_batch):
        """
        [新函数] 
        只对 prompt (不含 response) 运行 V-model，并返回一个标量 logit。
        """
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs
            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            # 这些是 prompt_only 的输入
            input_ids = micro_batch["prompt_input_ids"]
            attention_mask = micro_batch["prompt_attention_mask"]
            position_ids = micro_batch["prompt_position_ids"]
            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)

            # 假设 V-model (critic_module) 运行
            output = self.critic_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
                output_hidden_states=True, # 确保我们能拿到 hidden states
            )

            # 从 TRL 的 AutoModelForCausalLMWithValueHead 结构中获取 v_head 的输出
            if hasattr(self.critic_module, "v_head"):
                # (batch_size, prompt_seq_len, hidden_dim)
                hidden_states = output.hidden_states[-1] 
                # (batch_size, prompt_seq_len, 1)
                values_seq = self.critic_module.v_head(hidden_states)

                # 我们需要最后一个 non-padded token 的 value
                if attention_mask is not None:
                    # (batch_size,)
                    last_token_indices = attention_mask.sum(1) - 1 
                    # (batch_size,)
                    prompt_value_logit = values_seq[
                        torch.arange(values_seq.shape[0], device=values_seq.device), last_token_indices
                    ].squeeze(-1)
                else:
                    # (batch_size,)
                    prompt_value_logit = values_seq[:, -1].squeeze(-1)

            else:
                # 备用逻辑：如果 v_head 只是 logits
                logits = output.logits # (batch_size, prompt_seq_len, 1)
                if attention_mask is not None:
                    last_token_indices = attention_mask.sum(1) - 1
                    prompt_value_logit = logits[
                        torch.arange(logits.shape[0], device=logits.device), last_token_indices
                    ].squeeze(-1)
                else:
                    prompt_value_logit = logits[:, -1].squeeze(-1)
            
            # 返回形状为 (batch_size,) 的标量 logits
            return prompt_value_logit
        
    def _forward_micro_batch(self, micro_batch):
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                    )

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.critic_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating

                if hasattr(self.critic_module, "v_head"):
                    # For trl.AutoModelForCausalLMWithValueHead
                    values_rmpad = output[2].squeeze(0).unsqueeze(-1)
                else:
                    values_rmpad = output.logits
                    values_rmpad = values_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    values_rmpad = gather_outputs_and_unpad(
                        values_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )

                # pad it back
                values = pad_input(values_rmpad, indices=indices, batch=batch, seqlen=seqlen).squeeze(-1)
                values = values[:, -response_length - 1 : -1]
            else:
                output = self.critic_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                if hasattr(self.critic_module, "v_head"):
                    # For trl.AutoModelForCausalLMWithValueHead
                    values = output[2]
                else:
                    values = output.logits
                values = values[:, -response_length - 1 : -1].squeeze(-1)
            return values

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.critic_module, FSDP):
            grad_norm = self.critic_module.clip_grad_norm_(self.config.grad_clip)
        elif isinstance(self.critic_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.critic_optimizer.zero_grad()
        else:
            self.critic_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp critic", logger=logger)
    def compute_values(self, data: DataProto) -> torch.Tensor:
        self.critic_module.eval()
        
        # --- 新逻辑：准备 prompt-only 输入 ---
        response_length = data.batch["responses"].size(-1)
        # 假设 response 总是在 input_ids 的末尾
        prompt_input_ids = data.batch["input_ids"][:, :-response_length]
        prompt_attention_mask = data.batch["attention_mask"][:, :-response_length]
        
        pos_ids = data.batch["position_ids"]
        if pos_ids.dim() == 3: # (c, b, s)
            prompt_position_ids = pos_ids[:, :, :-response_length]
        else: # (b, s)
            prompt_position_ids = pos_ids[:, :-response_length]

        prompt_data_batch = {
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "prompt_position_ids": prompt_position_ids,
        }
        current_batch_size = prompt_input_ids.shape[0] 
        # 2. 将 dict 转换为 TensorDict
        prompt_data_batch_td = TensorDict(
            source=prompt_data_batch,
            batch_size=[current_batch_size]
        )
        prompt_data_non_tensor = {}
        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            prompt_data_non_tensor["multi_modal_inputs"] = data.non_tensor_batch["multi_modal_inputs"]
        
        prompt_data = DataProto(
            batch=prompt_data_batch_td,
            non_tensor_batch=prompt_data_non_tensor,
            meta_info=data.meta_info
        )
        # --- 结束新逻辑 ---

        micro_batch_size = data.meta_info["micro_batch_size"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        
        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(prompt_data, max_token_len=max_token_len)
        else:
            micro_batches = prompt_data.split(micro_batch_size)

        values_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                # 调用新的 prompt-only forward 函数
                values = self._forward_prompt_value(model_inputs) # (micro_batch_size)
            values_lst.append(values)
        
        # values.shape = (batch_size,)
        values = torch.concat(values_lst, dim=0)

        if use_dynamic_bsz:
            values = restore_dynamic_batch(values, batch_idx_list)

        # 返回 V(s_prompt) logits
        return values

    @GPUMemoryLogger(role="dp critic", logger=logger)
    def update_critic(self, data: DataProto):
        self.critic_module.train()
        metrics = {}

        # --- 修改 select_keys ---
        # "values" 是旧的 V(s_prompt) logits, 形状 [bs,]
        # "returns" 是目标 R (1或0), 形状 [bs,]
        select_keys = [
            "input_ids", "responses", "response_mask", "attention_mask", "position_ids",
            "values",  
            "returns" 
        ]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        mini_batches = data.split(self.config.ppo_mini_batch_size)

        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                
                # --- 准备 prompt-only 输入 ---
                response_length = mini_batch.batch["responses"].size(-1)
                prompt_input_ids = mini_batch.batch["input_ids"][:, :-response_length]
                prompt_attention_mask = mini_batch.batch["attention_mask"][:, :-response_length]
                pos_ids = mini_batch.batch["position_ids"]
                if pos_ids.dim() == 3:
                    prompt_position_ids = pos_ids[:, :, :-response_length]
                else:
                    prompt_position_ids = pos_ids[:, :-response_length]

                prompt_mb_batch = {
                    "prompt_input_ids": prompt_input_ids,
                    "prompt_attention_mask": prompt_attention_mask,
                    "prompt_position_ids": prompt_position_ids,
                }
                prompt_mb_non_tensor = {}
                if has_multi_modal_inputs:
                    prompt_mb_non_tensor["multi_modal_inputs"] = mini_batch.non_tensor_batch["multi_modal_inputs"]

                # 目标 R (1或0)
                R_targets = mini_batch.batch["returns"] # 形状 [mini_batch_size]
                # 1. 获取 batch size
                current_batch_size = prompt_input_ids.shape[0]
                # 2. 将 dict 转换为 TensorDict
                prompt_mb_batch_td = TensorDict(
                    source=prompt_mb_batch,
                    batch_size=[current_batch_size]
                )
                prompt_micro_batch_data = DataProto(
                    batch=prompt_mb_batch_td,
                    non_tensor_batch=prompt_mb_non_tensor,
                    meta_info=mini_batch.meta_info
                )
                
                if self.config.use_dynamic_bsz:
                    # ... (动态批处理逻辑)
                    # 警告: 动态批处理可能需要更复杂的 R_targets 分割
                    raise NotImplementedError("动态批处理与序列级V-loss尚未实现")
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    # 分割目标 R
                    R_targets_micro_batches = R_targets.split(self.config.ppo_micro_batch_size_per_gpu)
                    prompt_micro_batches = prompt_micro_batch_data.split(self.config.ppo_micro_batch_size_per_gpu)

                self.critic_optimizer.zero_grad()

                for micro_batch_idx, micro_batch in enumerate(prompt_micro_batches):
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

                    # 获取对应的 R 目标
                    R_targets_micro = R_targets_micro_batches[micro_batch_idx].to(get_device_id())
                    R_targets_micro_float = R_targets_micro.float()

                    # --- 新的 V-Loss 计算 ---
                    
                    # 1. 获取新的 V(s_prompt) logits
                    vpreds_logits = self._forward_prompt_value(model_inputs) # 形状 [micro_bs]
                    
                    # 2. 计算 BCE loss
                    # vpreds_logits: [micro_bs] (logits)
                    # R_targets_micro_float: [micro_bs] (labels, 0.0 or 1.0)
                    vf_loss_per_sample = self.bce_loss_fn(vpreds_logits, R_targets_micro_float)
                    
                    # 3. 聚合 loss
                    vf_loss = vf_loss_per_sample.mean()

                    # (我们不再使用 cliprange_value，所以 clipfrac 为 0)
                    vf_clipfrac = torch.tensor(0.0, device=vf_loss.device)
                    
                    # --- 结束新的 V-Loss 计算 ---
                    
                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = micro_batch.batch["prompt_input_ids"].shape[0] / self.config.ppo_mini_batch_size
                        loss = vf_loss * loss_scale_factor
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation
                        loss = vf_loss * loss_scale_factor

                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "critic/vf_loss": vf_loss.detach().item() * loss_scale_factor,
                            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
                            # 记录预测的平均成功率
                            "critic/vpred_prob_mean": torch.sigmoid(vpreds_logits).mean().detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)
                
                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"critic/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)

        self.critic_optimizer.zero_grad()
        return metrics