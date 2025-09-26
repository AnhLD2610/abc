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

"""Custom GRPO Trainer with contrastive learning and InfoNCE loss."""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from trl import GRPOTrainer
from transformers import PreTrainedModel
from .custom_rewards import get_contrastive_pairs


class ContrastiveGRPOTrainer(GRPOTrainer):
    """
    Extended GRPO Trainer with supervised contrastive loss and InfoNCE loss.
    """
    
        def __init__(
            self,
            model: PreTrainedModel,
            reward_funcs: List,
            contrastive_weight: float = 0.5,
            infonce_weight: float = 0.3,
            temperature: float = 0.07,
            **kwargs
        ):
            super().__init__(model=model, reward_funcs=reward_funcs, **kwargs)
            self.contrastive_weight = contrastive_weight
            self.infonce_weight = infonce_weight
            self.temperature = temperature
        
    def compute_contrastive_loss(
        self,
        hidden_states: torch.Tensor,
        prompt_end_positions: torch.Tensor,
        completion_end_positions: torch.Tensor,
        rewards: torch.Tensor,
        correctness: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            hidden_states: Last hidden states from model [batch_size, seq_len, hidden_dim]
            prompt_end_positions: Positions where prompts end [batch_size]
            completion_end_positions: Positions where completions end [batch_size]
            rewards: Reward values [batch_size]
            correctness: Binary correctness labels [batch_size]  
            lengths: Length of completions [batch_size]
            
        Returns:
            Contrastive loss
        """
        batch_size = hidden_states.size(0)
        
        # Get anchor embeddings (last token of prompt)
        anchor_embeds = []
        for i, pos in enumerate(prompt_end_positions):
            anchor_embeds.append(hidden_states[i, pos])
        anchor_embeds = torch.stack(anchor_embeds)  # [batch_size, hidden_dim]
        
        # Get completion embeddings (last token of completion)
        completion_embeds = []
        for i, pos in enumerate(completion_end_positions):
            completion_embeds.append(hidden_states[i, pos])
        completion_embeds = torch.stack(completion_embeds)  # [batch_size, hidden_dim]
        
        # Find positive and negative pairs
        positive_mask = correctness.bool()
        negative_mask = ~positive_mask
        
        if not positive_mask.any():
            # No positive examples: use highest reward + shortest length as positive
            combined_score = rewards - 0.1 * lengths / lengths.max()  # Prefer shorter
            best_idx = combined_score.argmax()
            positive_mask[best_idx] = True
            negative_mask[best_idx] = False
            
        if not negative_mask.any():
            # No negative examples: use longest correct answers as negatives
            if positive_mask.sum() > 1:
                correct_lengths = lengths * positive_mask.float() + (1 - positive_mask.float()) * lengths.min()
                longest_correct = correct_lengths.argmax()
                if positive_mask[longest_correct]:
                    negative_mask[longest_correct] = True
                    positive_mask[longest_correct] = False
        
        # Compute contrastive loss
        # Normalize embeddings
        anchor_embeds = F.normalize(anchor_embeds, dim=-1)
        completion_embeds = F.normalize(completion_embeds, dim=-1)
        
        # Compute similarities
        similarities = torch.matmul(anchor_embeds, completion_embeds.T) / self.temperature
        
        # Create labels: positive pairs get label 1, negative pairs get label 0
        labels = torch.zeros(batch_size, batch_size, device=similarities.device)
        
        for i in range(batch_size):
            if positive_mask[i]:
                # Find best positive for this anchor (shortest correct answer)
                if positive_mask.sum() > 0:
                    positive_lengths = lengths * positive_mask.float() + (1 - positive_mask.float()) * float('inf')
                    best_positive = positive_lengths.argmin()
                    labels[i, best_positive] = 1
        
        # Supervised contrastive loss with multiple positives
        loss = 0.0
        num_valid = 0
        
        for i in range(batch_size):
            if positive_mask[i]:  # Only compute loss for samples that have positives
                pos_similarities = similarities[i][positive_mask]
                neg_similarities = similarities[i][negative_mask] if negative_mask.any() else torch.tensor([], device=similarities.device)
                
                if len(pos_similarities) > 0:
                    # Log-sum-exp of positives
                    pos_exp = torch.exp(pos_similarities)
                    
                    # All similarities (positive + negative)
                    all_similarities = similarities[i]
                    all_exp = torch.exp(all_similarities)
                    
                    # Contrastive loss: -log(sum(pos_exp) / sum(all_exp))
                    loss += -torch.log(pos_exp.sum() / all_exp.sum())
                    num_valid += 1
        
        return loss / max(num_valid, 1)
    
    def compute_infonce_loss(
        self,
        hidden_states: torch.Tensor,
        prompt_end_positions: torch.Tensor,
        completion_end_positions: torch.Tensor,
        correctness: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss with one positive and multiple negatives.
        
        Args:
            hidden_states: Last hidden states from model
            prompt_end_positions: Positions where prompts end
            completion_end_positions: Positions where completions end
            correctness: Binary correctness labels
            lengths: Length of completions
            
        Returns:
            InfoNCE loss
        """
        batch_size = hidden_states.size(0)
        
        # Get embeddings
        anchor_embeds = []
        for i, pos in enumerate(prompt_end_positions):
            anchor_embeds.append(hidden_states[i, pos])
        anchor_embeds = torch.stack(anchor_embeds)
        
        completion_embeds = []
        for i, pos in enumerate(completion_end_positions):
            completion_embeds.append(hidden_states[i, pos])
        completion_embeds = torch.stack(completion_embeds)
        
        # Find the best positive (correct + shortest)
        positive_mask = correctness.bool()
        
        if not positive_mask.any():
            return torch.tensor(0.0, device=hidden_states.device)
        
        # Get shortest correct answer as the single positive
        correct_lengths = lengths * positive_mask.float() + (1 - positive_mask.float()) * float('inf')
        best_positive_idx = correct_lengths.argmin()
        
        # Normalize embeddings
        anchor_embeds = F.normalize(anchor_embeds, dim=-1)
        completion_embeds = F.normalize(completion_embeds, dim=-1)
        
        # Compute InfoNCE loss for each anchor
        total_loss = 0.0
        num_valid = 0
        
        for i in range(batch_size):
            anchor = anchor_embeds[i]  # [hidden_dim]
            
            # Positive: the best positive completion
            positive = completion_embeds[best_positive_idx]  # [hidden_dim]
            pos_sim = torch.dot(anchor, positive) / self.temperature
            
            # Negatives: all other completions
            neg_indices = [j for j in range(batch_size) if j != best_positive_idx]
            if len(neg_indices) > 0:
                negatives = completion_embeds[neg_indices]  # [num_neg, hidden_dim]
                neg_sims = torch.matmul(anchor.unsqueeze(0), negatives.T) / self.temperature  # [1, num_neg]
                neg_sims = neg_sims.squeeze(0)  # [num_neg]
                
                # InfoNCE: -log(exp(pos_sim) / (exp(pos_sim) + sum(exp(neg_sims))))
                all_sims = torch.cat([pos_sim.unsqueeze(0), neg_sims])  # [1 + num_neg]
                loss = -pos_sim + torch.logsumexp(all_sims, dim=0)
                
                total_loss += loss
                num_valid += 1
        
        return total_loss / max(num_valid, 1)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override compute_loss to add contrastive and InfoNCE losses.
        """
        # Get base GRPO loss
        if return_outputs:
            base_loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        else:
            base_loss = super().compute_loss(model, inputs, return_outputs=False)
            outputs = None
        
        # Extract necessary information for contrastive learning
        if hasattr(inputs, 'input_ids') and hasattr(inputs, 'attention_mask'):
            with torch.no_grad():
                model_outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    output_hidden_states=True
                )
                hidden_states = model_outputs.hidden_states[-1]  # Last layer
            
            # Get positions and rewards (these would need to be passed in inputs)
            if all(key in inputs for key in ['prompt_end_positions', 'completion_end_positions', 'rewards', 'correctness', 'lengths']):
                contrastive_loss = self.compute_contrastive_loss(
                    hidden_states=hidden_states,
                    prompt_end_positions=inputs['prompt_end_positions'],
                    completion_end_positions=inputs['completion_end_positions'],
                    rewards=inputs['rewards'],
                    correctness=inputs['correctness'],
                    lengths=inputs['lengths'],
                )
                
                infonce_loss = self.compute_infonce_loss(
                    hidden_states=hidden_states,
                    prompt_end_positions=inputs['prompt_end_positions'],
                    completion_end_positions=inputs['completion_end_positions'],
                    correctness=inputs['correctness'],
                    lengths=inputs['lengths'],
                )
                
                # Combine losses
                total_loss = base_loss + self.contrastive_weight * contrastive_loss + self.infonce_weight * infonce_loss
                
                # Log losses
                self.log({
                    'train/grpo_loss': base_loss.item(),
                    'train/contrastive_loss': contrastive_loss.item(),
                    'train/infonce_loss': infonce_loss.item(),
                    'train/total_loss': total_loss.item(),
                })
                
                if return_outputs:
                    return total_loss, outputs
                else:
                    return total_loss
        
        # Fallback to base loss if contrastive learning info not available
        if return_outputs:
            return base_loss, outputs
        else:
            return base_loss


### hàm có thể dùng trong grpo trainer 
'''
def _get_last_hidden_state(
        self,
        unwrapped_model,
        input_ids,
        attention_mask,
        logits_to_keep,
        pixel_values=None,
        image_grid_thw=None,
        pixel_attention_mask=None,
        image_sizes=None,
    ):
'''


'''
def get_high_entropy_mask(self, entropies: torch.Tensor, mask: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Returns a binary mask identifying tokens whose entropy exceeds a given quantile threshold.
'''

'''
def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
    ) -> dict[str, Optional[torch.Tensor]]:
        """Compute log-probs and (optionally) entropies for each token."""
'''

# này để tính advantage 
'''
    def _generate_and_score_completions(
'''

'''
def compute_liger_loss(self, unwrapped_model, inputs):
        # Compute the per-token log probabilities for the model
'''

'''
@profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        if self.use_liger_loss:
            # Compute the loss using the liger grpo loss
            unwrapped_model = self.accelerator.unwrap_model(model)
            return self._forward_redirection(model, unwrapped_model, self.compute_liger_loss, unwrapped_model, inputs)
        else:
            return self._compute_loss(model, inputs)

    def _compute_loss(self, model, inputs):
'''