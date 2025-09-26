# Efficient Reasoning với Enhanced GRPO

Đây là implementation của các contribute để giải quyết vấn đề blackbox GRPO trainer và cải thiện efficient reasoning.

## ✨ Tính Năng Chính

### 1. Enhanced GRPO Trainer với Group-wise Contrastive Learning

**Vấn đề**: GRPO trainer gốc là blackbox không thể modify trực tiếp.

**Giải pháp**: Tạo `EnhancedGRPOTrainer` kế thừa hoàn toàn từ `GRPOTrainer` với:

- **🔄 Backward Compatible**: Hoàn toàn tương thích với GRPO gốc
- **🎯 Group-wise Contrastive**: Contrastive learning trong từng nhóm N generations từ cùng 1 prompt
- **📊 InfoNCE per Group**: Single best positive vs negatives trong cùng generation group  
- **⚡ Seamless Integration**: Tự động enable/disable dựa trên config

**Architecture:**
```
1 prompt → N generations → N embeddings + 1 prompt embedding
├── Supervised Contrastive: All correct answers = positive, wrong answers = negative  
└── InfoNCE: Best correct answer = positive, others = negative
```

```python
# Trong enhanced_trainer.py
class EnhancedGRPOTrainer(GRPOTrainer):
    def _compute_contrastive_loss(self, ...):
        # Per-group contrastive: all correct answers as positives
        for prompt_idx in range(num_prompts):
            positive_mask = rewards[prompt_idx] > 0  # All correct = positive
            # Contrastive loss within this generation group
        
    def _compute_infonce_loss(self, ...):
        # Per-group InfoNCE: single best positive
        for prompt_idx in range(num_prompts):
            best_positive = find_best_correct_shortest(group)
            # InfoNCE loss within this generation group
```

### 2. Length + Accuracy Reward

**Mục tiêu**: Gen ra câu trả lời đúng + ngắn gọn.

**Implementation**: 

```python
def length_accuracy_reward(completions, solution, **kwargs):
    # Correct answers: reward ∝ 1/length (shorter = better)
    # Wrong answers: penalty ∝ length (longer = worse penalty)
    if is_correct:
        reward = 0.5 + 0.5 * (max_len - length) / (max_len - min_len)
    else:
        reward = -0.5 - 0.5 * (length - min_len) / (max_len - min_len)
```

### 3. Xử Lý Edge Cases

**Case 1**: Không có positive nào
- **Giải pháp**: Chọn sample có reward cao nhất + length ngắn nhất làm positive

**Case 2**: Không có negative nào  
- **Giải pháp**: Dùng majority voting (TTRL paper) để tạo consensus label

```python
def majority_voting_reward(completions, solution, **kwargs):
    # Đếm tần suất xuất hiện của mỗi đáp án
    answer_counts = Counter(all_answers)
    consensus_answer = answer_counts.most_common(1)[0][0]
    # Reward dựa vào consensus + length
```

### 4. High Entropy Exploration

**Vấn đề**: Câu hỏi khó không sample được đáp án đúng.

**Giải pháp**: Dùng entropy exploration (paper REASONING WITH EXPLORATION)

```python
def entropy_exploration_reward(completions, solution, difficulty_threshold=0.3):
    success_rate = correct_count / total_count
    if success_rate < difficulty_threshold:  # Câu khó
        # Add entropy bonus để khuyến khích exploration
        reward += entropy_bonus
    # Câu dễ train bình thường
```

## 🚀 Cách Sử Dụng

### 1. Quick Start với Config:

```yaml
# Enhanced GRPO with Contrastive Learning
contrastive_weight: 0.5    # Set > 0 to enable contrastive loss
infonce_weight: 0.3        # Set > 0 to enable InfoNCE loss
temperature: 0.07          # Temperature for similarity computation

# Custom reward functions
reward_funcs:
  - "length_accuracy"      # Combined length + accuracy reward
  - "majority_voting"      # Handle cases with no ground truth
  - "entropy_exploration"  # High entropy for difficult questions
```

### 2. Chạy Training:

```bash
# Với contrastive learning
python -m open_r1.grpo --config example_config.yaml

# Hoặc disable contrastive learning (fallback to standard GRPO)
python -m open_r1.grpo --config standard_config.yaml \
  --contrastive_weight 0.0 --infonce_weight 0.0
```

### 3. Programmatic Usage:

```python
from open_r1.enhanced_trainer import EnhancedGRPOTrainer

# Automatically chooses Enhanced or Standard based on weights
trainer = EnhancedGRPOTrainer(
    model=model,
    reward_funcs=["length_accuracy", "entropy_exploration"],
    contrastive_weight=0.5,      # > 0 enables contrastive learning
    infonce_weight=0.3,          # > 0 enables InfoNCE
    contrastive_temperature=0.07,
    enable_length_reward_scaling=True,
    **standard_grpo_args
)
```

## Kiến Trúc Implementation

```
open_r1/
├── custom_rewards.py          # Custom reward functions
├── custom_trainer.py          # Extended GRPO trainer
├── data_collator.py          # Data collator với position tracking
├── grpo.py                   # Updated main script
├── configs.py                # Updated với custom parameters
└── example_config.yaml       # Example configuration
```

## Các Reward Functions Mới

1. **`length_accuracy`**: Combined length + accuracy reward
2. **`majority_voting`**: Majority voting cho consensus labels
3. **`entropy_exploration`**: High entropy exploration cho câu khó

## Lợi Ích

1. **Giải quyết blackbox problem**: Extend thay vì modify GRPO trainer
2. **Efficient reasoning**: Câu trả lời đúng + ngắn gọn
3. **Handle edge cases**: Majority voting khi không có label
4. **Adaptive exploration**: High entropy cho câu khó, bình thường cho câu dễ
5. **Flexible configuration**: Dễ dàng config và experiment

## Technical Details

- **Contrastive Loss**: Supervised contrastive với multiple positives/negatives
- **InfoNCE Loss**: Cross-entropy với 1 positive + nhiều negatives  
- **Position Tracking**: Track prompt end và completion end positions
- **Reward Integration**: Combine với existing GRPO rewards
- **Memory Efficient**: Không duplicate data, chỉ thêm metadata

Với implementation này, bạn có thể train model để generate efficient reasoning responses trong khi vẫn leverage được GRPO framework hiện có.
