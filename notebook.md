# Fine-Tuning DeepSeek-R1 for Mathematical Reasoning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/your-username/your-gist-id)

This guide provides a comprehensive walkthrough for fine-tuning the DeepSeek-R1 model on mathematical reasoning tasks

## Table of Contents
- [Overview](#overview)
- [Key Components](#key-components)
- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Model Configuration](#model-configuration)
- [Training](#training)
- [Inference](#inference)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)

## Overview
DeepSeek-R1 is a state-of-the-art reasoning model optimized for complex problem-solving. This tutorial focuses on adapting it for mathematical reasoning tasks using the GSM8K dataset, which contains grade school math problems with step-by-step solutions.

## Key Components
1. **Model**: `deepseek-ai/deepseek-r1` (7B parameter version)
2. **Dataset**: GSM8K (8.5K high-quality math problems)
3. **Techniques**:
   - Gradient Checkpointing (memory optimization)
   - Mixed Precision Training (FP16)
   - Question-Answer Formatting

## Setup

### Hardware Requirements
- GPU with ≥16GB VRAM (NVIDIA A100/V100 recommended)
- 20GB+ Disk Space

### Environment Setup
```python
!pip install transformers==4.36.0
!pip install datasets==2.14.6
!pip install accelerate==0.25.0
!pip install torch==2.1.0
```

## Data Preparation

### Dataset Loading
```python
from datasets import load_dataset

# Load GSM8K dataset
dataset = load_dataset("gsm8k", "main", split="train")

# Split dataset (90% train, 10% validation)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
```

### Preprocessing
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-r1")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

def format_qa(example):
    return {
        "text": f"Solve this problem step-by-step:\nQuestion: {example['question']}\nAnswer: {example['answer']}"
    }

# Apply formatting and tokenization
dataset = dataset.map(format_qa).map(
    lambda examples: tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ),
    batched=True
)
```

## Model Configuration

### Model Initialization
```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-r1",
    device_map="auto",
    torch_dtype=torch.float16,
    use_cache=False  # Required for gradient checkpointing
)

# Enable memory optimizations
model.gradient_checkpointing_enable()
```

## Training

### Training Arguments
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./deepseek-r1-finetuned",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=True,
    report_to="none"
)
```

### Trainer Setup
```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer
)

# Start training
trainer.train()
```

## Inference

### Generation Configuration
```python
from transformers import GenerationConfig

gen_config = GenerationConfig(
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    do_sample=True,
    max_new_tokens=256,
    pad_token_id=tokenizer.eos_token_id
)
```

### Sample Inference
```python
def generate_response(question):
    prompt = f"Solve this problem step-by-step:\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        generation_config=gen_config
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
question = "A bookstore sold 45 books on Monday and twice as many on Tuesday. Each book costs $15. What's the total revenue?"
print(generate_response(question))
```

## Customization

### Dataset Formatting
For custom datasets, maintain this structure:
```python
{
    "question": "There are 15 apples and 12 oranges. If 8 fruits are sold, how many remain?",
    "answer": "1. Total fruits = 15 + 12 = 27\n2. Remaining fruits = 27 - 8 = 19\nFinal answer: 19"
}
```

### Hyperparameter Tuning
| Parameter          | Recommended Range | Description                |
|--------------------|-------------------|----------------------------|
| `learning_rate`    | 1e-5 to 5e-5      | Lower rates for fine-tuning|
| `per_device_batch` | 1-4               | Adjust based on GPU memory |
| `temperature`      | 0.5-1.0           | Higher = more creative     |

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**:
   - Reduce batch size
   - Enable `gradient_checkpointing`
   - Use `fp16` or `bf16` precision

2. **Poor Convergence**:
   - Try different learning rates
   - Increase training epochs
   - Verify dataset formatting

3. **Generation Quality**:
   - Adjust temperature and top-p values
   - Increase `max_new_tokens`
   - Add repetition penalty

## Resources
- [Original Model Card](https://huggingface.co/deepseek-ai/deepseek-r1)
- [GSM8K Paper](https://arxiv.org/abs/2110.14168)
- [Hugging Face Documentation](https://huggingface.co/docs)
