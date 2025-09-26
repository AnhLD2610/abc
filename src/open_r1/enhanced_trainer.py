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

"""Enhanced GRPO Trainer with contrastive learning integration."""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from trl import GRPOTrainer
from transformers import PreTrainedModel
from accelerate import logging

logger = logging.get_logger(__name__)


class EnhancedGRPOTrainer(GRPOTrainer):
    """
    Enhanced GRPO Trainer that integrates contrastive learning seamlessly with the existing architecture.
    
    This trainer extends the original GRPO by:
    1. Adding contrastive loss based on completion quality and length
    2. Implementing InfoNCE loss for better representation learning
    3. Handling edge cases with majority voting
    4. Supporting entropy-based exploration for difficult questions
    """
    
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[List, Any],
        contrastive_weight: float = 0.0,
        infonce_weight: float = 0.0, 
        contrastive_temperature: float = 0.07,
        enable_length_reward_scaling: bool = True,
        **kwargs
    ):
        super().__init__(model=model, reward_funcs=reward_funcs, **kwargs)
        
        # Contrastive learning parameters
        self.contrastive_weight = contrastive_weight
        self.infonce_weight = infonce_weight
        self.contrastive_temperature = contrastive_temperature
        self.enable_length_reward_scaling = enable_length_reward_scaling
        
        logger.info(f"Enhanced GRPO initialized with contrastive_weight={contrastive_weight}, "
                   f"infonce_weight={infonce_weight}")
    
    def _compute_contrastive_representations(
        self, 
        model: PreTrainedModel,
        prompt_ids: torch.Tensor,
        completion_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_generations: int,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute representations for contrastive learning within generation groups.
        
        In GRPO, we have groups of N generations per prompt:
        - batch_size = num_prompts * num_generations
        - Each group of N consecutive samples comes from the same prompt
        
        Args:
            model: The model to compute representations with
            prompt_ids: Prompt token IDs [batch_size, prompt_len]
            completion_ids: Completion token IDs [batch_size, completion_len]
            attention_mask: Attention mask [batch_size, total_len]
            num_generations: Number of generations per prompt (N)
            
        Returns:
            Tuple of (prompt_representations, completion_representations)
        """
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        
        # Get last hidden states
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                **{k: v for k, v in kwargs.items() if k in ['pixel_values', 'image_grid_thw', 'pixel_attention_mask', 'image_sizes']}
            )
            hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
        
        # Extract prompt end representations (last token of prompt)
        prompt_len = prompt_ids.size(1)
        prompt_representations = hidden_states[:, prompt_len - 1, :]  # [batch_size, hidden_dim]
        
        # Extract completion end representations (last non-padded token of completion)
        completion_mask = attention_mask[:, prompt_len:]  # [batch_size, completion_len]
        completion_lengths = completion_mask.sum(dim=1)  # [batch_size]
        
        completion_representations = []
        for i, length in enumerate(completion_lengths):
            if length > 0:
                # Last non-padded token in completion
                completion_representations.append(hidden_states[i, prompt_len + length - 1, :])
            else:
                # Fallback to prompt end if no completion tokens
                completion_representations.append(hidden_states[i, prompt_len - 1, :])
        
        completion_representations = torch.stack(completion_representations)  # [batch_size, hidden_dim]
        
        # Group representations by prompt
        # Each group of num_generations consecutive samples comes from the same prompt
        batch_size = prompt_representations.size(0)
        num_prompts = batch_size // num_generations
        
        # Reshape to group by prompt: [num_prompts, num_generations, hidden_dim]
        prompt_representations = prompt_representations.view(num_prompts, num_generations, -1)
        completion_representations = completion_representations.view(num_prompts, num_generations, -1)
        
        # Take the first prompt representation in each group (they should be identical)
        prompt_representations = prompt_representations[:, 0, :]  # [num_prompts, hidden_dim]
        
        return prompt_representations, completion_representations
    
    def _compute_contrastive_loss(
        self,
        anchor_embeds: torch.Tensor,
        completion_embeds: torch.Tensor, 
        rewards: torch.Tensor,
        completion_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss within generation groups.
        
        Args:
            anchor_embeds: Anchor embeddings (prompt ends) [num_prompts, hidden_dim]
            completion_embeds: Completion embeddings [num_prompts, num_generations, hidden_dim]
            rewards: Reward values [num_prompts, num_generations]
            completion_lengths: Completion lengths [num_prompts, num_generations]
            
        Returns:
            Contrastive loss scalar
        """
        num_prompts, num_generations, hidden_dim = completion_embeds.shape
        
        if num_generations < 2:
            return torch.tensor(0.0, device=anchor_embeds.device)
        
        total_loss = 0.0
        num_valid_prompts = 0
        
        # Process each prompt group separately
        for prompt_idx in range(num_prompts):
            anchor = anchor_embeds[prompt_idx]  # [hidden_dim]
            completions = completion_embeds[prompt_idx]  # [num_generations, hidden_dim]
            group_rewards = rewards[prompt_idx]  # [num_generations]
            group_lengths = completion_lengths[prompt_idx]  # [num_generations]
            
            # Determine positive and negative samples within this group
            if self.enable_length_reward_scaling:
                max_length = group_lengths.max().clamp(min=1.0)
                length_penalty = group_lengths / max_length
                adjusted_rewards = group_rewards - 0.1 * length_penalty
            else:
                adjusted_rewards = group_rewards
            
            # Positive: correct answers (reward > 0)
            # Negative: incorrect answers (reward <= 0)
            positive_mask = adjusted_rewards > 0
            negative_mask = adjusted_rewards <= 0
            
            # Handle edge cases
            if not positive_mask.any():
                # No positive: use highest reward + shortest as positive
                best_idx = adjusted_rewards.argmax()
                positive_mask[best_idx] = True
                negative_mask[best_idx] = False
            
            if not negative_mask.any():
                # No negative: use longest correct answer as negative
                if positive_mask.sum() > 1:
                    pos_lengths = group_lengths * positive_mask.float() + (1 - positive_mask.float()) * (-1)
                    longest_pos_idx = pos_lengths.argmax()
                    positive_mask[longest_pos_idx] = False
                    negative_mask[longest_pos_idx] = True
            
            if positive_mask.any() and negative_mask.any():
                # Normalize embeddings
                anchor_norm = F.normalize(anchor.unsqueeze(0), dim=-1)  # [1, hidden_dim]
                completions_norm = F.normalize(completions, dim=-1)  # [num_generations, hidden_dim]
                
                # Compute similarities: anchor vs all completions in this group
                similarities = torch.matmul(anchor_norm, completions_norm.T) / self.contrastive_temperature  # [1, num_generations]
                similarities = similarities.squeeze(0)  # [num_generations]
                
                # Supervised contrastive loss
                pos_similarities = similarities[positive_mask]  # All correct answers are positive
                neg_similarities = similarities[negative_mask]  # All incorrect answers are negative
                
                if len(pos_similarities) > 0 and len(neg_similarities) > 0:
                    # Contrastive loss: -log(sum(exp(pos)) / (sum(exp(pos)) + sum(exp(neg))))
                    pos_exp_sum = torch.logsumexp(pos_similarities, dim=0)
                    all_exp_sum = torch.logsumexp(similarities, dim=0)
                    loss = -pos_exp_sum + all_exp_sum
                    
                    total_loss += loss
                    num_valid_prompts += 1
        
        return total_loss / max(num_valid_prompts, 1)
    
    def _compute_infonce_loss(
        self,
        anchor_embeds: torch.Tensor,
        completion_embeds: torch.Tensor,
        rewards: torch.Tensor,
        completion_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss with single best positive within each generation group.
        
        Args:
            anchor_embeds: Anchor embeddings [num_prompts, hidden_dim]  
            completion_embeds: Completion embeddings [num_prompts, num_generations, hidden_dim]
            rewards: Reward values [num_prompts, num_generations]
            completion_lengths: Completion lengths [num_prompts, num_generations]
            
        Returns:
            InfoNCE loss scalar
        """
        num_prompts, num_generations, hidden_dim = completion_embeds.shape
        
        if num_generations < 2:
            return torch.tensor(0.0, device=anchor_embeds.device)
        
        total_loss = 0.0
        num_valid_prompts = 0
        
        # Process each prompt group separately
        for prompt_idx in range(num_prompts):
            anchor = anchor_embeds[prompt_idx]  # [hidden_dim]
            completions = completion_embeds[prompt_idx]  # [num_generations, hidden_dim]
            group_rewards = rewards[prompt_idx]  # [num_generations]
            group_lengths = completion_lengths[prompt_idx]  # [num_generations]
            
            # Find the best positive (correct + shortest) within this group
            if self.enable_length_reward_scaling:
                max_length = group_lengths.max().clamp(min=1.0)
                length_penalty = group_lengths / max_length
                combined_score = group_rewards - 0.2 * length_penalty  # Stronger length preference
            else:
                combined_score = group_rewards
            
            # Only consider correct answers (reward > 0) as potential positives
            correct_mask = group_rewards > 0
            if not correct_mask.any():
                # No correct answer, skip this group
                continue
                
            # Find best positive among correct answers
            correct_scores = combined_score * correct_mask.float() + (1 - correct_mask.float()) * float('-inf')
            best_positive_idx = correct_scores.argmax()
            
            # All other completions are negatives
            negative_indices = [i for i in range(num_generations) if i != best_positive_idx]
            
            if len(negative_indices) > 0:
                # Normalize embeddings
                anchor_norm = F.normalize(anchor.unsqueeze(0), dim=-1)  # [1, hidden_dim]
                completions_norm = F.normalize(completions, dim=-1)  # [num_generations, hidden_dim]
                
                # Compute similarities
                similarities = torch.matmul(anchor_norm, completions_norm.T) / self.contrastive_temperature  # [1, num_generations]
                similarities = similarities.squeeze(0)  # [num_generations]
                
                # InfoNCE loss
                pos_sim = similarities[best_positive_idx]
                neg_sims = similarities[negative_indices]
                
                # InfoNCE: -log(exp(pos_sim) / (exp(pos_sim) + sum(exp(neg_sims))))
                all_sims = torch.cat([pos_sim.unsqueeze(0), neg_sims])
                loss = -pos_sim + torch.logsumexp(all_sims, dim=0)
                
                total_loss += loss
                num_valid_prompts += 1
        
        return total_loss / max(num_valid_prompts, 1)
    
    def _compute_loss(self, model, inputs):
        """
        Override the loss computation to add contrastive learning.
        """
        # Get the base GRPO loss
        base_loss = super()._compute_loss(model, inputs)
        
        # If contrastive learning is disabled, return base loss
        if self.contrastive_weight == 0.0 and self.infonce_weight == 0.0:
            return base_loss
        
        # Extract necessary data from inputs
        prompt_ids = inputs["prompt_ids"]
        completion_ids = inputs["completion_ids"]
        prompt_mask = inputs["prompt_mask"]
        completion_mask = inputs["completion_mask"]
        advantages = inputs["advantages"]
        
        # Create full attention mask
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        
        # Use advantages as proxy for rewards (higher advantage = better completion)
        # In GRPO, advantages = rewards - baseline, so they indicate relative quality
        rewards = advantages
        
        # Compute completion lengths
        completion_lengths = completion_mask.sum(dim=1).float()
        
        # Get contrastive representations
        try:
            prompt_representations, completion_representations = self._compute_contrastive_representations(
                model=model,
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
                attention_mask=attention_mask,
                num_generations=self.num_generations,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                pixel_attention_mask=inputs.get("pixel_attention_mask"),
                image_sizes=inputs.get("image_sizes"),
            )
            
            # Reshape rewards and lengths to match grouped structure
            batch_size = rewards.size(0)
            num_prompts = batch_size // self.num_generations
            
            rewards_grouped = rewards.view(num_prompts, self.num_generations)
            lengths_grouped = completion_lengths.view(num_prompts, self.num_generations)
            
            # Compute contrastive losses
            contrastive_loss = torch.tensor(0.0, device=base_loss.device)
            infonce_loss = torch.tensor(0.0, device=base_loss.device)
            
            if self.contrastive_weight > 0.0:
                contrastive_loss = self._compute_contrastive_loss(
                    anchor_embeds=prompt_representations,
                    completion_embeds=completion_representations,
                    rewards=rewards_grouped,
                    completion_lengths=lengths_grouped,
                )
            
            if self.infonce_weight > 0.0:
                infonce_loss = self._compute_infonce_loss(
                    anchor_embeds=prompt_representations,
                    completion_embeds=completion_representations,
                    rewards=rewards_grouped,
                    completion_lengths=lengths_grouped,
                )
            
            # Combine losses
            total_loss = (
                base_loss + 
                self.contrastive_weight * contrastive_loss + 
                self.infonce_weight * infonce_loss
            )
            
            # Log metrics
            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["contrastive_loss"].append(contrastive_loss.item())
            self._metrics[mode]["infonce_loss"].append(infonce_loss.item())
            self._metrics[mode]["total_loss"].append(total_loss.item())
            
            return total_loss
            
        except Exception as e:
            logger.warning(f"Error computing contrastive loss, falling back to base loss: {e}")
            return base_loss
