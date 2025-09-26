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

"""Data collator for contrastive GRPO training with position tracking."""

import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from transformers import DataCollatorWithPadding
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@dataclass
class ContrastiveGRPODataCollator:
    """
    Data collator for GRPO training with contrastive learning support.
    
    This collator tracks prompt end positions and completion end positions
    needed for contrastive learning.
    """
    
    tokenizer: PreTrainedTokenizerBase
    padding: bool = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate features and add position information for contrastive learning.
        
        Args:
            features: List of features from dataset
            
        Returns:
            Batch dictionary with additional position and reward information
        """
        batch_size = len(features)
        
        # Extract basic inputs
        input_ids = []
        attention_masks = []
        prompt_end_positions = []
        completion_end_positions = []
        rewards = []
        correctness = []
        lengths = []
        
        for feature in features:
            # Get input_ids and attention_mask
            if 'input_ids' in feature:
                input_ids.append(torch.tensor(feature['input_ids']))
                attention_masks.append(torch.tensor(feature['attention_mask']))
            
            # Get position information
            if 'prompt_end_position' in feature:
                prompt_end_positions.append(feature['prompt_end_position'])
            else:
                # Fallback: try to find prompt end by looking for user/assistant tokens
                input_id_list = feature.get('input_ids', [])
                # This is a simple heuristic - you might need to adjust based on your tokenizer
                prompt_end_pos = len(input_id_list) // 2  # Rough estimate
                prompt_end_positions.append(prompt_end_pos)
            
            if 'completion_end_position' in feature:
                completion_end_positions.append(feature['completion_end_position'])
            else:
                # Use the last non-padding token position
                input_id_list = feature.get('input_ids', [])
                completion_end_positions.append(len(input_id_list) - 1)
            
            # Get reward and correctness information
            rewards.append(feature.get('reward', 0.0))
            correctness.append(feature.get('is_correct', False))
            
            # Get completion length
            if 'completion_length' in feature:
                lengths.append(feature['completion_length'])
            else:
                # Estimate completion length
                prompt_end = prompt_end_positions[-1]
                completion_end = completion_end_positions[-1]
                lengths.append(max(0, completion_end - prompt_end))
        
        # Pad sequences
        if input_ids:
            # Pad input_ids and attention_masks
            max_len = max(len(ids) for ids in input_ids)
            if self.max_length:
                max_len = min(max_len, self.max_length)
            
            padded_input_ids = []
            padded_attention_masks = []
            
            for i, (ids, mask) in enumerate(zip(input_ids, attention_masks)):
                # Truncate if necessary
                if len(ids) > max_len:
                    ids = ids[:max_len]
                    mask = mask[:max_len]
                    # Adjust positions if truncated
                    prompt_end_positions[i] = min(prompt_end_positions[i], max_len - 1)
                    completion_end_positions[i] = min(completion_end_positions[i], max_len - 1)
                
                # Pad
                pad_length = max_len - len(ids)
                padded_ids = torch.cat([ids, torch.full((pad_length,), self.tokenizer.pad_token_id)])
                padded_mask = torch.cat([mask, torch.zeros(pad_length)])
                
                padded_input_ids.append(padded_ids)
                padded_attention_masks.append(padded_mask)
            
            batch = {
                'input_ids': torch.stack(padded_input_ids),
                'attention_mask': torch.stack(padded_attention_masks),
                'prompt_end_positions': torch.tensor(prompt_end_positions, dtype=torch.long),
                'completion_end_positions': torch.tensor(completion_end_positions, dtype=torch.long),
                'rewards': torch.tensor(rewards, dtype=torch.float),
                'correctness': torch.tensor(correctness, dtype=torch.bool),
                'lengths': torch.tensor(lengths, dtype=torch.float),
            }
        else:
            # Fallback for features without input_ids
            batch = {
                'prompt_end_positions': torch.tensor(prompt_end_positions, dtype=torch.long),
                'completion_end_positions': torch.tensor(completion_end_positions, dtype=torch.long), 
                'rewards': torch.tensor(rewards, dtype=torch.float),
                'correctness': torch.tensor(correctness, dtype=torch.bool),
                'lengths': torch.tensor(lengths, dtype=torch.float),
            }
        
        return batch


def prepare_contrastive_data(
    examples: List[Dict[str, Any]], 
    tokenizer: PreTrainedTokenizerBase,
    reward_funcs: List,
) -> List[Dict[str, Any]]:
    """
    Prepare data with contrastive learning information.
    
    Args:
        examples: List of examples from dataset
        tokenizer: Tokenizer to use
        reward_funcs: List of reward functions to compute rewards
        
    Returns:
        List of examples with additional contrastive learning info
    """
    prepared_examples = []
    
    for example in examples:
        # Tokenize prompt and completion
        prompt_text = example.get('prompt', '')
        completion_text = example.get('completion', '')
        
        # Tokenize prompt only
        prompt_tokens = tokenizer(prompt_text, add_special_tokens=True, return_tensors="pt")
        prompt_end_position = prompt_tokens['input_ids'].shape[1] - 1
        
        # Tokenize full text (prompt + completion)
        full_text = prompt_text + completion_text
        full_tokens = tokenizer(full_text, add_special_tokens=True, return_tensors="pt", truncation=True)
        completion_end_position = full_tokens['input_ids'].shape[1] - 1
        
        # Compute rewards using reward functions
        if reward_funcs and 'solution' in example:
            # Mock completion format for reward function
            mock_completions = [[{"content": completion_text}]]
            mock_solutions = [example['solution']]
            
            rewards = []
            for reward_func in reward_funcs:
                try:
                    reward_values = reward_func(mock_completions, mock_solutions)
                    if reward_values and reward_values[0] is not None:
                        rewards.append(reward_values[0])
                except Exception as e:
                    print(f"Error computing reward: {e}")
                    rewards.append(0.0)
            
            # Use average reward
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            is_correct = avg_reward > 0.0
        else:
            avg_reward = 0.0
            is_correct = False
        
        # Prepare example with contrastive info
        prepared_example = {
            **example,
            'input_ids': full_tokens['input_ids'].squeeze().tolist(),
            'attention_mask': full_tokens['attention_mask'].squeeze().tolist(),
            'prompt_end_position': prompt_end_position,
            'completion_end_position': completion_end_position,
            'reward': avg_reward,
            'is_correct': is_correct,
            'completion_length': len(completion_text),
        }
        
        prepared_examples.append(prepared_example)
    
    return prepared_examples
