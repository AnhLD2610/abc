from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen3-0.6B"

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# 2 prompts có độ dài khác nhau
messages = [
    [{"role": "user", "content": "Hi!"}],
    [{"role": "user", "content": "Give me a short introduction to large language models and their impact."}]
]

texts = [
    tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    for msg in messages
]

# Encode batch
model_inputs = tokenizer(
    texts,
    return_tensors="pt",
    padding=True,           # pad input trong batch
    truncation=True,
    max_length=512          # giới hạn để dễ xem
).to(model.device)

print("Input shape:", model_inputs["input_ids"].shape)  # (batch, seq_len)

# Generate batch
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=30000,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id
)
print(generated_ids) 
# Tách completion cho từng prompt
# for i, text in enumerate(texts):
#     input_len = (model_inputs["input_ids"][i] != tokenizer.pad_token_id).sum().item()
#     completion_ids = generated_ids[i, input_len:]
#     print(f"\n=== Prompt {i+1} ===")
#     print("Input length:", input_len)
#     print("Completion length:", completion_ids.shape[0])
#     print("Completion IDs:", completion_ids.tolist())
#     print("Completion text:", tokenizer.decode(completion_ids, skip_special_tokens=True))
