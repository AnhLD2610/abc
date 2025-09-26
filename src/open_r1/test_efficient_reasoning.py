#!/usr/bin/env python3
"""
Test script for Efficient Reasoning with Enhanced GRPO
"""

import torch
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer
from enhanced_trainer import EnhancedGRPOTrainer
from trl import GRPOConfig

def create_dummy_reward_function():
    """Create a dummy reward function for testing"""
    def reward_fn(prompts, completions, **kwargs):
        rewards = []
        for prompt, completion in zip(prompts, completions):
            # Simple reward: longer completions get higher reward, with some randomness
            base_reward = len(completion.split()) / 50.0
            # Add correctness simulation (random for testing)
            import random
            correctness = random.choice([0.0, 1.0])  # 50% chance of being "correct"
            
            # Combine accuracy (0.7 weight) and length penalty (0.3 weight)
            length_penalty = max(0.1, 1.0 - len(completion.split()) / 100.0)  # Prefer shorter
            final_reward = 0.7 * correctness + 0.3 * length_penalty
            
            rewards.append(final_reward)
        return rewards
    
    return reward_fn

def test_enhanced_grpo():
    """Test the Enhanced GRPO Trainer"""
    print("🚀 Testing Enhanced GRPO with Contrastive Learning...")
    
    # Load config
    with open("efficient_reasoning_config.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    
    # Create GRPO config
    config = GRPOConfig(
        output_dir=config_dict["output_dir"],
        num_train_epochs=1,  # Short test
        per_device_train_batch_size=2,  # Small batch for testing
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=config_dict["learning_rate"],
        num_generations=config_dict["num_generations"],
        temperature=config_dict["temperature"],
        top_p=config_dict["top_p"],
        max_prompt_length=256,  # Shorter for testing
        max_completion_length=128,
        beta=config_dict["beta"],
        epsilon=config_dict["epsilon"],
        logging_steps=1,
        eval_steps=5,
        save_steps=10,
        bf16=False,  # Use fp32 for testing
        remove_unused_columns=False,
    )
    
    # Create dummy dataset
    dummy_data = [
        {"prompt": "What is 2 + 2?"},
        {"prompt": "Solve: x + 3 = 7"},
        {"prompt": "What is the capital of France?"},
        {"prompt": "Calculate 5 * 6"},
    ] * 4  # Repeat to have enough data
    
    from datasets import Dataset
    dataset = Dataset.from_list(dummy_data)
    
    # Create reward function
    reward_fn = create_dummy_reward_function()
    
    # Initialize Enhanced GRPO Trainer
    trainer = EnhancedGRPOTrainer(
        model=config_dict["model_name_or_path"],
        reward_funcs=reward_fn,
        args=config,
        train_dataset=dataset,
        eval_dataset=dataset[:8],  # Small eval set
        
        # Contrastive learning parameters
        contrastive_weight=config_dict["contrastive_weight"],
        infonce_weight=config_dict["infonce_weight"],
        contrastive_temperature=config_dict["contrastive_temperature"],
        enable_length_reward_scaling=config_dict["enable_length_reward_scaling"],
    )
    
    print("✅ Enhanced GRPO Trainer initialized successfully!")
    print(f"📊 Contrastive Weight: {trainer.contrastive_weight}")
    print(f"📊 InfoNCE Weight: {trainer.infonce_weight}")
    print(f"🌡️  Contrastive Temperature: {trainer.contrastive_temperature}")
    print(f"📏 Length Reward Scaling: {trainer.enable_length_reward_scaling}")
    
    # Test a few steps
    print("\n🔥 Running test training steps...")
    try:
        trainer.train()
        print("✅ Training completed successfully!")
        
        # Print some metrics
        if hasattr(trainer, '_metrics') and 'train' in trainer._metrics:
            metrics = trainer._metrics['train']
            print("\n📈 Training Metrics:")
            for key, values in metrics.items():
                if values:
                    avg_val = sum(values) / len(values)
                    print(f"  {key}: {avg_val:.4f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 Test completed!")

if __name__ == "__main__":
    test_enhanced_grpo()