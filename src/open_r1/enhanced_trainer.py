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

"""Enhanced GRPO Trainer with contrastive learning for efficient reasoning."""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import Counter
from trl import GRPOTrainer
from transformers import PreTrainedModel
from accelerate import logging

## reward 
from rewards import accuracy_reward, len_reward, format_reward
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from collections import Counter


logger = logging.get_logger(__name__)


class EnhancedGRPOTrainer(GRPOTrainer):
    """
    Enhanced GRPO Trainer implementing:
    1. Supervised Contrastive Learning with length + accuracy rewards
    2. InfoNCE loss for shortest correct answers
    3. Majority voting for edge cases (TTRL paper)
    4. High entropy exploration for hard questions (REASONING WITH EXPLORATION paper)
    """
    
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[List, Any],
        use_contrastive: bool = False,
        use_infonce: bool = False,
        contrastive_weight: float = 0.1,
        infonce_weight: float = 0.1,
        contrastive_temperature: float = 0.07,
        length_reward_weight: float = 0.3,
        accuracy_reward_weight: float = 0.7,
        high_entropy_temperature: float = 1.5,
        **kwargs
    ):
        super().__init__(model=model, reward_funcs=reward_funcs, **kwargs)
        
        # Contrastive learning settings
        self.use_contrastive = use_contrastive
        self.use_infonce = use_infonce
        self.contrastive_weight = contrastive_weight
        self.infonce_weight = infonce_weight
        self.contrastive_temperature = contrastive_temperature
        
        # Reward composition settings
        self.length_reward_weight = length_reward_weight
        self.accuracy_reward_weight = accuracy_reward_weight
        
        # Entropy exploration settings
        self.high_entropy_temperature = high_entropy_temperature
        
        logger.info(f"Enhanced GRPO initialized: contrastive={use_contrastive}, "
                   f"infonce={use_infonce}, weights=({contrastive_weight}, {infonce_weight})")


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


    def _majority_voting_fallback(
        self, 
        completions: List[List[Dict[str, str]]], 
        solution: List[str], 
        **kwargs
    ) -> Tuple[int, List[float]]:
        """
        Apply majority voting when no positive samples exist (TTRL paper approach).
        
        Khi tất cả N samples đều sai, dùng majority voting để tìm đáp án xuất hiện nhiều nhất,
        coi đó là đáp án "đúng" mới, và tính lại rewards dựa trên đáp án consensus này.
        
        Args:
            completions: List of completion dicts (format như GRPO)
            solution: List of ground truth solutions
            
        Returns:
            consensus_idx: Index of consensus completion
            new_rewards: Updated rewards based on consensus answer
        """
        
        if not completions:
            return 0, []
        
        # Extract content from completions
        contents = [completion[0]["content"] for completion in completions]
        
        # Parse all answers and check if any are originally correct
        parsed_answers = []
        original_rewards = []
        
        for content, sol in zip(contents, solution):
            # Parse ground truth
            gold_parsed = parse(sol, extraction_mode="first_match")
            
            if len(gold_parsed) != 0:
                # Parse completion answer with strict latex requirements
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed="all",
                                units=True,
                            ),
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                
                parsed_answers.append(answer_parsed[0] if answer_parsed else None)
                
                # Check if originally correct
                if answer_parsed and len(answer_parsed) > 0:
                    is_correct = verify(gold_parsed[0], answer_parsed[0])
                    original_rewards.append(1.0 if is_correct else 0.0)
                else:
                    original_rewards.append(0.0)
            else:
                parsed_answers.append(None)
                original_rewards.append(0.0)
        
        # If any answer is actually correct, no need for majority voting
        if any(r > 0 for r in original_rewards):
            best_idx = original_rewards.index(max(original_rewards))
            return best_idx, original_rewards
        
        # All answers are wrong - apply majority voting on parsed answers
        valid_parsed = [ans for ans in parsed_answers if ans is not None]
        
        if not valid_parsed:
            # No valid answers, return first with all zeros
            return 0, original_rewards
        
        # Count frequencies of parsed answers
        answer_counts = Counter(valid_parsed)
        max_count = max(answer_counts.values())
        
        # Find most frequent answer(s)
        consensus_candidates = [ans for ans, count in answer_counts.items() if count == max_count]
        
        # If tie, choose shortest answer
        if len(consensus_candidates) > 1:
            consensus_answer = min(consensus_candidates, key=len)
        else:
            consensus_answer = consensus_candidates[0]
        
        # Find index of consensus answer
        consensus_idx = 0
        for i, parsed_ans in enumerate(parsed_answers):
            if parsed_ans == consensus_answer:
                consensus_idx = i
                break
        
        # Recalculate rewards: consensus answer = 1.0, others = 0.0
        new_rewards = []
        for parsed_ans in parsed_answers:
            if parsed_ans is not None and parsed_ans == consensus_answer:
                new_rewards.append(1.0)  # Treat consensus as "correct"
            else:
                new_rewards.append(0.0)  # All others as "incorrect"
        
        return consensus_idx, new_rewards

    def _get_pos_neg_masks(
        self, 
        rewards: torch.Tensor, 
        lengths: torch.Tensor, 
        completions: Optional[List[List[Dict[str, str]]]] = None,
        solution: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get positive/negative masks for a group of completions.
        Handles edge cases with majority voting.
        
        Args:
            rewards: [num_generations] - accuracy rewards
            lengths: [num_generations] - completion lengths
            completions: Optional list of completion strings for majority voting
            
        Returns:
            pos_mask, neg_mask: Boolean tensors indicating positive/negative samples
        """
        num_generations = len(rewards)
        
        # Combine accuracy and length for final rewards
        # Normalize lengths to [0, 1] range
        max_length = lengths.max().clamp(min=1.0)
        length_penalty = lengths / max_length
        
        # Final reward = accuracy_weight * accuracy + length_weight * (1 - length_penalty)
        final_rewards = (self.accuracy_reward_weight * rewards + 
                        self.length_reward_weight * (1 - length_penalty))
        
        # Initial positive/negative classification
        pos_mask = rewards == 1  # Correct answers
        neg_mask = rewards != 1  # Incorrect answers
        
        # Edge case 1: No positives - fallback to best final reward + shortest length
        # (Majority voting đã được thực hiện sớm hơn trong _generate_and_score_completions)
        if not pos_mask.any():
            # Fallback: choose best final reward + shortest length
            best_reward = final_rewards.max()
            best_indices = (final_rewards == best_reward).nonzero(as_tuple=True)[0]
            if len(best_indices) > 1:
                # Among best, choose shortest
                best_lengths = lengths[best_indices]
                shortest_idx = best_indices[best_lengths.argmin()]
            else:
                shortest_idx = best_indices[0]
            
            pos_mask = torch.zeros(num_generations, dtype=torch.bool, device=rewards.device)
            pos_mask[shortest_idx] = True
            neg_mask = ~pos_mask
        
        # Edge case 2: No negatives - convert longest correct to negative
        elif not neg_mask.any() and pos_mask.sum() > 1:
            pos_lengths = lengths * pos_mask.float() + (1 - pos_mask.float()) * (-1)
            longest_pos_idx = pos_lengths.argmax()
            pos_mask[longest_pos_idx] = False
            neg_mask[longest_pos_idx] = True
        
        return pos_mask, neg_mask

    def _get_contrastive_embeddings(
        self, 
        model: PreTrainedModel,
        prompt_ids: torch.Tensor,
        completion_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get embeddings for contrastive learning by forward passing prompt+completion.
        
        The key insight: Generated completions can't be backpropped through, but we can
        forward pass prompt+completion to get embeddings for contrastive learning.
        
        Args:
            model: The model
            prompt_ids: [batch_size, prompt_len]
            completion_ids: [batch_size, completion_len] 
            attention_mask: [batch_size, total_len]
            
        Returns:
            prompt_embeds: [batch_size, hidden_dim] - embeddings of last prompt token
            completion_embeds: [batch_size, hidden_dim] - embeddings of last completion token (before EOS)
        """
        # Concatenate prompt + completion
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        
        # Forward pass to get hidden states
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False
            )
            hidden_states = outputs.hidden_states[-1]  # Last layer: [batch_size, seq_len, hidden_dim]
        
        # Extract embeddings
        prompt_len = prompt_ids.size(1)
        batch_size = input_ids.size(0)
        
        # Prompt embeddings: last token of prompt
        prompt_embeds = hidden_states[:, prompt_len - 1, :]  # [batch_size, hidden_dim]
        
        # Completion embeddings: last non-padded token of completion (before EOS)
        completion_embeds = []
        for i in range(batch_size):
            # Find last non-padded position in completion
            completion_mask = attention_mask[i, prompt_len:]
            last_pos = completion_mask.sum() - 1  # -1 because EOS is the last token
            if last_pos >= 0:
                completion_embeds.append(hidden_states[i, prompt_len + last_pos, :])
            else:
                # Fallback to prompt end if no completion
                completion_embeds.append(hidden_states[i, prompt_len - 1, :])
        
        completion_embeds = torch.stack(completion_embeds)  # [batch_size, hidden_dim]
        
        return prompt_embeds, completion_embeds

    def _compute_supervised_contrastive_loss(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor, 
        negatives: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss: multiple positives vs multiple negatives.
        
        Args:
            anchors: [num_prompts, hidden_dim]
            positives: [num_prompts, num_pos, hidden_dim] 
            negatives: [num_prompts, num_neg, hidden_dim]
            
        Returns:
            Contrastive loss scalar
        """
        num_prompts = anchors.size(0)
        total_loss = 0.0
        num_valid = 0
        
        for i in range(num_prompts):
            anchor = F.normalize(anchors[i], dim=-1)  # [hidden_dim]
            pos = F.normalize(positives[i], dim=-1)   # [num_pos, hidden_dim]
            neg = F.normalize(negatives[i], dim=-1)   # [num_neg, hidden_dim]
            
            if pos.size(0) > 0 and neg.size(0) > 0:
                # Compute similarities
                pos_sims = torch.matmul(anchor, pos.T) / self.contrastive_temperature  # [num_pos]
                neg_sims = torch.matmul(anchor, neg.T) / self.contrastive_temperature  # [num_neg]
                
                # Supervised contrastive loss
                pos_exp_sum = torch.logsumexp(pos_sims, dim=0)
                all_sims = torch.cat([pos_sims, neg_sims])
                all_exp_sum = torch.logsumexp(all_sims, dim=0)
                
                loss = -pos_exp_sum + all_exp_sum
                total_loss += loss
                num_valid += 1
        
        return total_loss / max(num_valid, 1)

    def _compute_infonce_loss(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss: single shortest correct answer vs all others.
        
        Args:
            anchors: [num_prompts, hidden_dim]
            positives: [num_prompts, hidden_dim] - single shortest correct answer
            negatives: [num_prompts, num_neg, hidden_dim] - all other answers
            
        Returns:
            InfoNCE loss scalar
        """
        num_prompts = anchors.size(0)
        total_loss = 0.0
        num_valid = 0
        
        for i in range(num_prompts):
            anchor = F.normalize(anchors[i], dim=-1)     # [hidden_dim]
            positive = F.normalize(positives[i], dim=-1) # [hidden_dim]
            neg = F.normalize(negatives[i], dim=-1)      # [num_neg, hidden_dim]
            
            if neg.size(0) > 0:
                # Compute similarities
                pos_sim = torch.dot(anchor, positive) / self.contrastive_temperature
                neg_sims = torch.matmul(anchor, neg.T) / self.contrastive_temperature  # [num_neg]
                
                # InfoNCE loss
                all_sims = torch.cat([pos_sim.unsqueeze(0), neg_sims])
                loss = -pos_sim + torch.logsumexp(all_sims, dim=0)
                
                total_loss += loss
                num_valid += 1
        
        return total_loss / max(num_valid, 1)

    def _generate_and_score_completions(self, model, inputs):
        """
        Override to apply majority voting for both GRPO and contrastive learning.
        Majority voting được áp dụng ngay sau generation để cả GRPO và contrastive đều dùng rewards đã cập nhật.
        """
        # Original generation - chỉ gọi 1 lần
        completions_dict = super()._generate_and_score_completions(model, inputs)
        
        # Extract data for majority voting
        rewards = completions_dict.get('advantages', completions_dict.get('rewards'))
        if rewards is None:
            return completions_dict
            
        # Get completions and solutions for majority voting
        completions = completions_dict.get('completions')  # Generated completions
        solution = inputs.get('solution', inputs.get('labels'))  # Ground truth solutions
        
        if completions is None or solution is None:
            return completions_dict
            
        # Reshape rewards to group by prompts
        batch_size = rewards.size(0)
        if batch_size % self.args.num_generations != 0:
            return completions_dict
            
        num_prompts = batch_size // self.args.num_generations
        rewards_grouped = rewards.view(num_prompts, self.args.num_generations)
        
        # Apply majority voting for groups with no positive samples
        updated_rewards = rewards.clone()
        num_majority_voted = 0
        
        for prompt_idx in range(num_prompts):
            start_idx = prompt_idx * self.args.num_generations
            end_idx = (prompt_idx + 1) * self.args.num_generations
            
            group_rewards = rewards_grouped[prompt_idx]
            group_completions = completions[start_idx:end_idx]
            group_solution = solution[start_idx:end_idx]  # Should be same for all in group
            
            # Check if this group has no positive samples
            if not (group_rewards == 1).any():
                # Apply majority voting
                try:
                    consensus_idx, new_rewards = self._majority_voting_fallback(
                        group_completions, group_solution
                    )
                    
                    # Update rewards for this group
                    updated_rewards[start_idx:end_idx] = torch.tensor(
                        new_rewards, device=rewards.device, dtype=rewards.dtype
                    )
                    
                    num_majority_voted += 1
                    
                except Exception as e:
                    logger.warning(f"Majority voting failed for group {prompt_idx}: {e}")
                    continue
        
        if num_majority_voted > 0:
            logger.info(f"Applied majority voting to {num_majority_voted}/{num_prompts} groups with no positive samples")
            
            # Update completions_dict with new rewards
            completions_dict['advantages'] = updated_rewards
            if 'rewards' in completions_dict:
                completions_dict['rewards'] = updated_rewards
        
        # Detect and log hard questions (after majority voting)
        rewards_grouped_updated = updated_rewards.view(num_prompts, self.args.num_generations)
        hard_mask = self._detect_hard_questions(rewards_grouped_updated)
        
        if hard_mask.any():
            logger.info(f"After majority voting: {hard_mask.sum().item()}/{num_prompts} questions still hard")
        
        return completions_dict

    def _compute_loss(self, model, inputs):
        """
        Override to add contrastive learning losses.
        """
        # Get base GRPO loss
        base_loss = super()._compute_loss(model, inputs)
        
        # Return early if contrastive learning disabled
        if not (self.use_contrastive or self.use_infonce):
            return base_loss
            
        try:
            # Extract data from inputs
            prompt_ids = inputs["prompt_ids"]
            completion_ids = inputs["completion_ids"] 
            prompt_mask = inputs["prompt_mask"]
            completion_mask = inputs["completion_mask"]
            advantages = inputs["advantages"]  # Use as proxy for rewards
            
            # Create attention mask
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            
            # Get embeddings via forward pass of prompt+completion
            prompt_embeds, completion_embeds = self._get_contrastive_embeddings(
                model, prompt_ids, completion_ids, attention_mask
            )
            
            # Compute completion lengths
            completion_lengths = completion_mask.sum(dim=1).float()
            
            # Group by prompts (each group of N generations from same prompt)
            batch_size = advantages.size(0)
            num_prompts = batch_size // self.args.num_generations
            
            if batch_size % self.args.num_generations != 0:
                logger.warning(f"Batch size {batch_size} not divisible by num_generations {self.args.num_generations}")
                return base_loss
            
            # Reshape to group format
            rewards_grouped = advantages.view(num_prompts, self.args.num_generations)
            lengths_grouped = completion_lengths.view(num_prompts, self.args.num_generations)
            prompt_embeds_grouped = prompt_embeds.view(num_prompts, self.args.num_generations, -1)[:, 0, :]  # Take first (they're identical)
            completion_embeds_grouped = completion_embeds.view(num_prompts, self.args.num_generations, -1)
            
            # Initialize losses
            contrastive_loss = torch.tensor(0.0, device=base_loss.device)
            infonce_loss = torch.tensor(0.0, device=base_loss.device)
            
            # Compute losses for each prompt group
            if self.use_contrastive or self.use_infonce:
                all_anchors = []
                all_positives_cont = []  # For supervised contrastive (multiple positives)
                all_negatives_cont = []
                all_positives_info = []  # For InfoNCE (single shortest positive)
                all_negatives_info = []
                
                for prompt_idx in range(num_prompts):
                    group_rewards = rewards_grouped[prompt_idx]
                    group_lengths = lengths_grouped[prompt_idx]
                    group_completions_embeds = completion_embeds_grouped[prompt_idx]
                    
                    # Get positive/negative masks
                    pos_mask, neg_mask = self._get_pos_neg_masks(group_rewards, group_lengths)
                    
                    if pos_mask.any() and neg_mask.any():
                        anchor = prompt_embeds_grouped[prompt_idx]
                        
                        # For supervised contrastive: all positives
                        positives = group_completions_embeds[pos_mask]
                        negatives = group_completions_embeds[neg_mask]
                        
                        all_anchors.append(anchor)
                        all_positives_cont.append(positives)
                        all_negatives_cont.append(negatives)
                        
                        # For InfoNCE: shortest positive only
                        pos_indices = pos_mask.nonzero(as_tuple=True)[0]
                        pos_lengths = group_lengths[pos_indices]
                        shortest_pos_idx = pos_indices[pos_lengths.argmin()]
                        shortest_positive = group_completions_embeds[shortest_pos_idx]
                        
                        # All others (including other positives) as negatives for InfoNCE
                        infonce_neg_mask = torch.ones(self.args.num_generations, dtype=torch.bool, device=group_rewards.device)
                        infonce_neg_mask[shortest_pos_idx] = False
                        infonce_negatives = group_completions_embeds[infonce_neg_mask]
                        
                        all_positives_info.append(shortest_positive)
                        all_negatives_info.append(infonce_negatives)
                
                # Compute losses
                if all_anchors and self.use_contrastive:
                    anchors_tensor = torch.stack(all_anchors)
                    # Pad positives and negatives to same length for batching
                    max_pos = max(pos.size(0) for pos in all_positives_cont) if all_positives_cont else 0
                    max_neg = max(neg.size(0) for neg in all_negatives_cont) if all_negatives_cont else 0
                    
                    if max_pos > 0 and max_neg > 0:
                        # Simple approach: compute loss for each group separately
                        total_cont_loss = 0.0
                        num_valid = 0
                        for i, (anchor, pos, neg) in enumerate(zip(all_anchors, all_positives_cont, all_negatives_cont)):
                            if pos.size(0) > 0 and neg.size(0) > 0:
                                group_loss = self._compute_supervised_contrastive_loss(
                                    anchor.unsqueeze(0), pos.unsqueeze(0), neg.unsqueeze(0)
                                )
                                total_cont_loss += group_loss
                                num_valid += 1
                        contrastive_loss = total_cont_loss / max(num_valid, 1)
                
                if all_anchors and self.use_infonce:
                    anchors_tensor = torch.stack(all_anchors)
                    positives_tensor = torch.stack(all_positives_info)
                    
                    # Compute InfoNCE loss for each group
                    total_info_loss = 0.0
                    num_valid = 0
                    for i, (anchor, pos, neg) in enumerate(zip(all_anchors, all_positives_info, all_negatives_info)):
                        if neg.size(0) > 0:
                            group_loss = self._compute_infonce_loss(
                                anchor.unsqueeze(0), pos.unsqueeze(0), neg.unsqueeze(0)
                            )
                            total_info_loss += group_loss
                            num_valid += 1
                    infonce_loss = total_info_loss / max(num_valid, 1)
            
            # Combine losses
            total_loss = (base_loss + 
                         self.contrastive_weight * contrastive_loss + 
                         self.infonce_weight * infonce_loss)
            
            # Log losses
            if hasattr(self, 'log'):
                self.log({
                    "train/base_loss": base_loss.item(),
                    "train/contrastive_loss": contrastive_loss.item(),
                    "train/infonce_loss": infonce_loss.item(),
                    "train/total_loss": total_loss.item(),
                })
            
            return total_loss
            
        except Exception as e:
            logger.warning(f"Error in contrastive loss computation: {e}")
            return base_loss