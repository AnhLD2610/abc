# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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


import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import Counter
from trl import GRPOTrainer
from transformers import PreTrainedModel
from accelerate import logging

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
logger = logging.get_logger(__name__)

class ContrastiveTrainer(GRPOTrainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[List, Any],
        use_contrastive: bool = False,
        contrastive_weight: float = 0.1,
        contrastive_temperature: float = 0.07,
        length_reward_weight: float = 0.3,
        accuracy_reward_weight: float = 0.7,
        high_entropy_temperature: float = 1.5,
        **kwargs
    ):
        super().__init__(model=model, reward_funcs=reward_funcs, **kwargs)
        # Contrastive learning settings
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        
        # Reward composition settings
        self.length_reward_weight = length_reward_weight
        self.accuracy_reward_weight = accuracy_reward_weight
        
        # Entropy exploration settings
        self.high_entropy_temperature = high_entropy_temperature
        
        logger.info(
            f"Enhanced GRPO initialized: contrastive={use_contrastive}, weight={contrastive_weight}"
        )
    
    def _detect_hard_questions(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Detect hard questions where no sample achieves correct answer.
        Simple voting-based approach: if all samples are wrong, it's a hard question.
        
        Args:
            rewards: [num_prompts, num_generations] - reward values
            
        Returns:
            hard_mask: [num_prompts] - True for hard questions
        """
        # Hard questions: no sample gets correct answer (all accuracy rewards != 1)
        hard_mask = (rewards == 1).sum(dim=1) == 0  # [num_prompts]
        
        return hard_mask

    def _get_prompt_completion_embeddings(
        self,
        model,
        prompt_ids: torch.Tensor,
        completion_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_mask: torch.Tensor,
        auxiliary_inputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return embeddings for the final prompt token and final completion token."""

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        print('attention_mask')
        print(attention_mask)
        print(attention_mask.shape)
        print('attention_mask')


        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "use_cache": False,
        }

        for key in ("pixel_values", "image_grid_thw", "pixel_attention_mask", "image_sizes"):
            value = auxiliary_inputs.get(key)
            if value is not None:
                model_inputs[key] = value

        outputs = model(**model_inputs)
        hidden_states = outputs.hidden_states[-1]

        # ta có hidden_states 
        # để lấu

        # input_ids lấy maxlength (BS, SEQ_LENGTH) => LẤY ĐC TOKEN INPUT THEO SEQ_LENGTH 
        # 

        prompt_lengths = prompt_mask.sum(dim=1)
        completion_lengths = completion_mask.sum(dim=1)

        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        prompt_end_indices = (prompt_lengths - 1).clamp(min=0)
        prompt_embeddings = hidden_states[batch_indices, prompt_end_indices]

        # Fix: Tính toán completion end indices chính xác hơn
        completion_end_indices = (prompt_lengths + completion_lengths - 1).clamp(
            min=prompt_end_indices, max=input_ids.size(1) - 1
        )
        completion_embeddings = hidden_states[batch_indices, completion_end_indices]

        return prompt_embeddings, completion_embeddings

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Call the parent implementation to get the base result
        result = super()._generate_and_score_completions(inputs)
        
        # Extract the needed information for our custom logic
        prompt_ids = result["prompt_ids"]
        completion_ids = result["completion_ids"] 
        prompt_mask = result["prompt_mask"]
        completion_mask = result["completion_mask"]

        # Add debug prints if needed
        print(f'DEBUG: prompt_ids.shape: {prompt_ids.shape}')
        print(f'DEBUG: completion_ids.shape: {completion_ids.shape}')
        print(f'DEBUG: num_generations: {self.num_generations}')
        
        # Add group_ids to track which samples belong to the same prompt (before shuffle)
        batch_size = prompt_ids.size(0)
        group_size = self.num_generations
        
        if batch_size % group_size == 0:
            # Create group IDs: [0,0,1,1,2,2,...] for num_generations=2
            group_ids = torch.arange(batch_size // group_size, device=prompt_ids.device)
            group_ids = group_ids.repeat_interleave(group_size)
            result["group_ids"] = group_ids
            print(f'DEBUG: Added group_ids: {group_ids}')
        else:
            logger.warning(f"Batch size {batch_size} not divisible by num_generations {group_size}")
            result["group_ids"] = None
        
        # Return the result from the parent class with added group_ids
        return result

    def _compute_loss(self, model, inputs):
        print("DEBUG: Entering _compute_loss")
        
        # Call the parent class loss computation
        base_loss = super()._compute_loss(model, inputs)
        
        # If not using contrastive learning, return the original loss
        if not self.use_contrastive:
            return base_loss
            
        print('DEBUG: Entering contrastive block')
        
        # Check if we have group information (from before shuffle)
        group_ids = inputs.get("group_ids")
        print('group_ids:')
        print(group_ids)
        print('group_ids.shape:')
        print(group_ids.shape)
        if group_ids is None:
            logger.warning("No group_ids found - skipping contrastive loss")
            return base_loss
        
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        advantages = inputs["advantages"]
        
        print("DEBUG: Starting contrastive embedding computation")
        print(f"DEBUG: group_ids after shuffle: {group_ids}")
        print(f"DEBUG: advantages after shuffle: {advantages}")
        
        try:
            # Compute embeddings for contrastive learning
            prompt_embeds, completion_embeds = self._get_prompt_completion_embeddings(
                model,
                prompt_ids,
                completion_ids,
                prompt_mask,
                completion_mask,
                {
                    "pixel_values": inputs.get("pixel_values"),
                    "image_grid_thw": inputs.get("image_grid_thw"),
                    "pixel_attention_mask": inputs.get("pixel_attention_mask"),
                    "image_sizes": inputs.get("image_sizes"),
                },
            )
            
            # Compute contrastive loss using group_ids to handle shuffle
            contrastive_loss = self._compute_contrastive_loss_with_groups(
                prompt_embeds, completion_embeds, advantages, group_ids
            )
            
            print(f"DEBUG: Base loss: {base_loss.item():.4f}, Contrastive loss: {contrastive_loss.item():.4f}")
            
            # Combine losses
            total_loss = base_loss + self.contrastive_weight * contrastive_loss
            
            # Log metrics
            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["contrastive_loss"].append(contrastive_loss.item())
            self._metrics[mode]["total_loss"].append(total_loss.item())
            
            return total_loss
            
        except Exception as e:
            logger.warning(f"Contrastive loss computation failed: {e}")
            return base_loss
    
    def _compute_contrastive_loss_with_groups(
        self, 
        prompt_embeds: torch.Tensor,
        completion_embeds: torch.Tensor,
        advantages: torch.Tensor,
        group_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss between good and bad completions for each prompt group.
        This version handles shuffled data by using group_ids to reconstruct groups.
        
        Args:
            prompt_embeds: [batch_size, hidden_dim] - embeddings for prompts
            completion_embeds: [batch_size, hidden_dim] - embeddings for completions
            advantages: [batch_size] - advantages for each completion
            group_ids: [batch_size] - group ID for each sample (same ID = same prompt)
            
        Returns:
            contrastive_loss: scalar loss value
        """
        device = prompt_embeds.device
        unique_groups = torch.unique(group_ids)
        total_loss = torch.tensor(0.0, device=device)
        valid_groups = 0
        
        for group_id in unique_groups:
            # Find all samples belonging to this group
            group_mask = (group_ids == group_id)
            group_indices = torch.where(group_mask)[0]
            
            if len(group_indices) < 2:
                continue  # Need at least 2 samples for contrastive learning
            
            # Get group data
            group_advantages = advantages[group_indices]
            group_completion_embeds = completion_embeds[group_indices]
            # Use the first prompt embedding as anchor (all should be same for same group)
            anchor_prompt_embed = prompt_embeds[group_indices[0]]
            
            # Skip if all advantages are the same (no contrast)
            if torch.allclose(group_advantages, group_advantages[0]):
                continue
                
            # Create positive and negative masks based on advantages
            pos_mask = group_advantages > group_advantages.mean()
            neg_mask = ~pos_mask
            
            # Skip if we don't have both positive and negative samples
            if not pos_mask.any() or not neg_mask.any():
                # Fallback: use best vs worst
                best_idx = torch.argmax(group_advantages)
                worst_idx = torch.argmin(group_advantages)
                if best_idx == worst_idx:
                    continue
                pos_mask = torch.zeros_like(group_advantages, dtype=torch.bool)
                neg_mask = torch.zeros_like(group_advantages, dtype=torch.bool)
                pos_mask[best_idx] = True
                neg_mask[worst_idx] = True
            
            # Compute similarities between prompt and completions
            similarities = torch.matmul(group_completion_embeds, anchor_prompt_embed) / self.contrastive_temperature
            
            # Compute contrastive loss using InfoNCE-style loss
            pos_similarities = similarities[pos_mask]
            neg_similarities = similarities[neg_mask]
            
            # For each positive sample, compute loss against all negative samples
            pos_loss = 0.0
            pos_count = 0
            
            for pos_sim in pos_similarities:
                # Create logits: [pos_sim, neg_sim1, neg_sim2, ...]
                logits = torch.cat([pos_sim.unsqueeze(0), neg_similarities])
                # Target is 0 (first element is positive)
                target = torch.zeros(1, dtype=torch.long, device=device)
                loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), target)
                pos_loss += loss
                pos_count += 1
            
            if pos_count > 0:
                group_loss = pos_loss / pos_count
                total_loss += group_loss
                valid_groups += 1
                
                print(f"DEBUG: Group {group_id.item()}: {len(group_indices)} samples, "
                      f"pos: {pos_mask.sum().item()}, neg: {neg_mask.sum().item()}, "
                      f"loss: {group_loss.item():.4f}")
        
        if valid_groups > 0:
            return total_loss / valid_groups
        else:
            return torch.tensor(0.0, device=device)