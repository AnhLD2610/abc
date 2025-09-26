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

"""Custom reward functions for efficient reasoning with length + accuracy rewards."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


def length_accuracy_reward(completions: List[List[Dict[str, str]]], solution: List[str], **kwargs) -> List[Optional[float]]:
    """
    Combined reward function that considers both accuracy and length for efficient reasoning.
    
    Args:
        completions: List of model completions  
        solution: List of ground truth solutions
        
    Returns:
        List of rewards where shorter correct answers get higher rewards
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    # First pass: check correctness
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            correctness.append(True)  # Skip unparseable examples
            print("Failed to parse gold solution: ", sol)
            continue
            
        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        try:
            is_correct = verify(answer_parsed, gold_parsed)
            correctness.append(is_correct)
        except Exception as e:
            print(f"Verification failed: {e}")
            correctness.append(False)
    
    # Second pass: calculate length-based rewards
    lengths = [len(content) for content in contents]
    min_len = min(lengths) if lengths else 1
    max_len = max(lengths) if lengths else 1
    
    for length, is_correct in zip(lengths, correctness):
        if is_correct:
            # For correct answers: reward inversely proportional to length
            # Shorter answers get higher rewards (0.5 to 1.0)
            if max_len == min_len:
                reward = 1.0
            else:
                reward = 0.5 + 0.5 * (max_len - length) / (max_len - min_len)
        else:
            # For incorrect answers: penalty proportional to length  
            # Longer wrong answers get more penalty (-1.0 to -0.5)
            if max_len == min_len:
                reward = -0.5
            else:
                reward = -0.5 - 0.5 * (length - min_len) / (max_len - min_len)
                
        rewards.append(float(reward))
    
    return rewards


def majority_voting_reward(completions: List[List[Dict[str, str]]], solution: List[str], **kwargs) -> List[Optional[float]]:
    """
    Reward function using majority voting for consensus labels when no ground truth is available.
    
    Args:
        completions: List of model completions
        solution: List of ground truth solutions (may be empty/None)
        
    Returns:
        List of rewards based on majority consensus
    """
    contents = [completion[0]["content"] for completion in completions]
    
    # Extract answers from all completions
    all_answers = []
    for content in contents:
        parsed_answers = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        
        if len(parsed_answers) > 0:
            all_answers.append(str(parsed_answers[0]))
        else:
            all_answers.append("")
    
    # Find consensus answer using majority voting
    if not all_answers:
        return [0.0] * len(completions)
    
    answer_counts = Counter(all_answers)
    consensus_answer = answer_counts.most_common(1)[0][0]
    
    # Assign rewards based on consensus
    rewards = []
    lengths = [len(content) for content in contents]
    min_len = min(lengths) if lengths else 1
    max_len = max(lengths) if lengths else 1
    
    for answer, length in zip(all_answers, lengths):
        if answer == consensus_answer and answer != "":
            # Correct answer: higher reward for shorter responses
            if max_len == min_len:
                reward = 1.0
            else:
                reward = 0.5 + 0.5 * (max_len - length) / (max_len - min_len)
        else:
            # Wrong or empty answer: penalty
            reward = -0.5
            
        rewards.append(float(reward))
    
    return rewards


def get_contrastive_pairs(
    completions: List[List[Dict[str, str]]], 
    solution: List[str], 
    rewards: List[float],
    **kwargs
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create anchor-positive-negative pairs for contrastive learning.
    
    Args:
        completions: List of model completions
        solution: List of ground truth solutions  
        rewards: List of rewards for each completion
        
    Returns:
        Tuple of (anchor_indices, positive_indices, negative_indices)
    """
    contents = [completion[0]["content"] for completion in completions]
    lengths = [len(content) for content in contents]
    
    # Determine correctness
    correctness = []
    for content, sol in zip(contents, solution):
        if not sol:  # If no ground truth, use majority voting
            correctness.append(rewards[contents.index(content)] > 0)
            continue
            
        gold_parsed = parse(sol, extraction_mode="first_match")
        if len(gold_parsed) == 0:
            correctness.append(True)
            continue
            
        answer_parsed = parse(content, extraction_mode="first_match")
        try:
            is_correct = verify(answer_parsed, gold_parsed)
            correctness.append(is_correct)
        except:
            correctness.append(False)
    
    # Find best positive (correct + shortest) and worst negative (wrong + longest)
    anchor_indices = []
    positive_indices = []  
    negative_indices = []
    
    correct_indices = [i for i, c in enumerate(correctness) if c]
    wrong_indices = [i for i, c in enumerate(correctness) if not c]
    
    if correct_indices:
        # Best positive: shortest correct answer
        best_positive = min(correct_indices, key=lambda i: lengths[i])
        positive_indices = [best_positive]
        
        # Use prompt end as anchor (we'll need to modify this in the trainer)
        anchor_indices = [best_positive]  # Placeholder
        
    if wrong_indices:
        # Negatives: all wrong answers
        negative_indices = wrong_indices
    elif correct_indices and len(correct_indices) > 1:
        # If no wrong answers, use longer correct answers as negatives
        sorted_correct = sorted(correct_indices, key=lambda i: lengths[i])
        negative_indices = sorted_correct[1:]  # All except the shortest
    
    return anchor_indices, positive_indices, negative_indices


def entropy_exploration_reward(
    completions: List[List[Dict[str, str]]], 
    solution: List[str],
    difficulty_threshold: float = 0.3,
    **kwargs
) -> List[Optional[float]]:
    """
    Reward function that encourages high entropy exploration for difficult questions.
    
    Args:
        completions: List of model completions
        solution: List of ground truth solutions
        difficulty_threshold: Threshold to determine if question is difficult
        
    Returns:
        List of rewards with entropy bonus for difficult questions
    """
    contents = [completion[0]["content"] for completion in completions]
    
    # First get base accuracy rewards
    base_rewards = length_accuracy_reward(completions, solution, **kwargs)
    
    # Calculate success rate as difficulty measure
    correct_count = sum(1 for r in base_rewards if r and r > 0)
    success_rate = correct_count / len(base_rewards) if base_rewards else 0
    
    # If success rate is low (difficult question), add entropy bonus
    if success_rate < difficulty_threshold:
        # For difficult questions, add small bonus to encourage exploration
        entropy_bonus = 0.1
        modified_rewards = []
        for reward in base_rewards:
            if reward is not None:
                modified_rewards.append(reward + entropy_bonus)
            else:
                modified_rewards.append(reward)
        return modified_rewards
    else:
        # For easy questions, use normal rewards
        return base_rewards
