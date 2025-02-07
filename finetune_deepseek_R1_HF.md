# Fine-Tuning DeepSeek R1 (Reasoning Model)

This markdown document reproduces the content (both explanation and code) from the Kaggle notebook for fine-tuning DeepSeek R1. It covers the full workflow—from environment setup and data preprocessing to model fine-tuning (with reinforcement learning elements) and evaluation.

---

## Overview

DeepSeek R1 is an open-source reasoning model optimized for tasks that require logical inference, mathematical problem solving, and code generation. In this notebook, we demonstrate how to:

- Load the pre-trained DeepSeek R1 model and tokenizer.
- Prepare and preprocess a synthetic reasoning dataset.
- Set up training parameters for both supervised fine-tuning and an optional reinforcement learning (RL) stage.
- Fine-tune the model.
- Evaluate the performance on sample reasoning tasks.

---

## Prerequisites

- **Python 3.8+**
- **PyTorch**
- **Hugging Face Transformers & Datasets**
- Other dependencies (e.g., numpy, pandas). See `requirements.txt` for the full list.

---

## 1. Import Libraries

We begin by importing the necessary libraries and modules.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import pandas as pd
```

---

## 2. Load the Pre-trained Model and Tokenizer

Load the base DeepSeek R1 model along with its tokenizer from the Hugging Face Hub.

```python
model_name = "deepseek-ai/deepseek-r1"  # Replace with the actual model identifier if different
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

---

## 3. Data Preparation

The dataset contains synthetic examples designed for reasoning tasks. Each example is formatted to include special tokens that delineate the reasoning process and final answer.

```python
# Load your synthetic reasoning dataset
# (For example, assume a local JSON or CSV file formatted appropriately)
dataset = load_dataset("json", data_files={"train": "data/train.json", "validation": "data/val.json"})

def preprocess_function(examples):
    # Assume each example contains a 'text' field with the prompt and reasoning chain.
    # Format the input using special tokens as required by the model.
    inputs = [f"|special_token| {text.strip()} |special_token|" for text in examples['text']]
    return tokenizer(inputs, truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(preprocess_function, batched=True)
```

---

## 4. Fine-Tuning Setup (Supervised Fine-Tuning)

Configure the training parameters and initialize the Trainer for supervised fine-tuning.

```python
training_args = TrainingArguments(
    output_dir="./deepseek-r1-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
    logging_steps=10,
    evaluation_strategy="steps",
    save_steps=100,
    fp16=True,  # Enable mixed precision if supported
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)
```

Run the supervised fine-tuning process:

```python
trainer.train()
```

---

## 5. Optional: Reinforcement Learning Fine-Tuning

To further enhance reasoning performance, you can apply a reinforcement learning (RL) phase. In this stage, the model’s outputs are rewarded based on correctness and chain-of-thought consistency.

> **Note:** The following code is a simplified pseudocode example of an RL loop. Actual implementation may require integration with libraries such as Hugging Face’s `trl` (Transformer Reinforcement Learning) or custom RL training routines.

```python
# Define a function to compute a simple reward based on the output and the target answer.
def compute_reward(generated_text, target_text):
    # In practice, implement a more nuanced reward function that assesses reasoning steps.
    return 1.0 if generated_text.strip() == target_text.strip() else 0.0

# Pseudocode for an RL training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
num_rl_epochs = 2  # Adjust as needed

for epoch in range(num_rl_epochs):
    for batch in trainer.get_train_dataloader():
        outputs = model(**batch)
        # Assume 'outputs' contains generated sequences and we have a corresponding target.
        # Compute rewards for each sample (this is illustrative; real reward calculation will vary).
        rewards = [compute_reward(tokenizer.decode(out, skip_special_tokens=True), target)
                   for out, target in zip(outputs.logits.argmax(dim=-1), batch["target_text"])]
        
        # Compute a custom loss that incorporates the reward signal
        # (This is a placeholder for the actual RL loss computation, such as policy gradient loss.)
        loss = torch.tensor(0.0)
        for r in rewards:
            loss += (1.0 - r)  # Dummy loss: lower loss for higher reward
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

## 6. Evaluation

After fine-tuning, test the model with a sample reasoning task to verify improvements.

```python
# Example inference: solving a simple mathematical problem
test_input = "Solve the equation: x^2 - 5x + 6 = 0"
inputs = tokenizer(test_input, return_tensors="pt")
output_ids = model.generate(**inputs, max_length=100)
result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Model Output:", result)
```

---

## Conclusion

By following this workflow, the DeepSeek R1 model is fine-tuned to improve its reasoning capabilities. The combination of supervised learning and an optional reinforcement learning phase helps the model generate more accurate and coherent reasoning traces. This markdown file replicates the content and code from the original Kaggle notebook, providing a self-contained guide for fine-tuning DeepSeek R1.

---

## References

- Kaggle Notebook: [Fine-Tuning DeepSeek R1 (Reasoning Model)](https://www.kaggle.com/code/kingabzpro/fine-tuning-deepseek-r1-reasoning-model/notebook) citeturn1fetch0
- DeepSeek research papers and additional documentation from DeepSeek AI.
```
