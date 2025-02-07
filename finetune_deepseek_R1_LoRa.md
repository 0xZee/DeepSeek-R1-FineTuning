# Fine-Tuning DeepSeek-V1.5 7B Model for Reasoning Tasks

This guide provides a step-by-step process to fine-tune the **DeepSeek-V1.5 7B** model on a reasoning dataset. The model is fine-tuned using the Hugging Face `transformers` library and the `peft` library for parameter-efficient fine-tuning.

---

## **Setup and Installation**

Before starting, ensure you have the necessary libraries installed:

```bash
!pip install -q transformers peft datasets accelerate bitsandbytes
```

---

## **Load the Pre-Trained Model and Tokenizer**

We will load the **DeepSeek-V1.5 7B** model and its tokenizer using the Hugging Face `transformers` library.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the pre-trained model and tokenizer
model_name = "deepseek-ai/deepseek-v1.5-7b"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

---

## **Prepare the Dataset**

The dataset used for fine-tuning is formatted for reasoning tasks. Each example includes a **prompt** and a **completion**.

### Example Dataset Format:
```json
[
    {
        "prompt": "What is the capital of France?",
        "completion": "The capital of France is Paris."
    },
    {
        "prompt": "Solve for x: 2x + 5 = 15",
        "completion": "To solve for x, subtract 5 from both sides: 2x = 10. Then divide by 2: x = 5."
    }
]
```

### Load and Tokenize the Dataset:
```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("your_dataset_name")  # Replace with your dataset path or name

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["prompt"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
```

---

## **Configure LoRA for Parameter-Efficient Fine-Tuning**

We use **LoRA (Low-Rank Adaptation)** for efficient fine-tuning. This reduces the number of trainable parameters while maintaining performance.

```python
from peft import LoraConfig, get_peft_model

# Define LoRA configuration
lora_config = LoraConfig(
    r=8,  # Rank of the low-rank matrices
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Target modules for LoRA
    lora_dropout=0.1,  # Dropout rate
    bias="none",  # Bias handling
    task_type="CAUSAL_LM"  # Task type
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
```

---

## **Training Setup**

We use the `Trainer` class from Hugging Face for training.

```python
from transformers import Trainer, TrainingArguments

# Define training arguments
training_args = TrainingArguments(
    output_dir="./deepseek-v1.5-7b-finetuned",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    save_total_limit=2,
    fp16=True,  # Enable mixed precision training
    push_to_hub=False  # Set to True if you want to push to Hugging Face Hub
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"]
)

# Start training
trainer.train()
```

---

## **Save the Fine-Tuned Model**

After training, save the fine-tuned model and tokenizer for future use.

```python
# Save the model and tokenizer
model.save_pretrained("./deepseek-v1.5-7b-finetuned")
tokenizer.save_pretrained("./deepseek-v1.5-7b-finetuned")
```

---

## **Inference with the Fine-Tuned Model**

You can now use the fine-tuned model for inference.

```python
# Load the fine-tuned model and tokenizer
model = AutoModelForCausalLM.from_pretrained("./deepseek-v1.5-7b-finetuned")
tokenizer = AutoTokenizer.from_pretrained("./deepseek-v1.5-7b-finetuned")

# Generate a response
input_text = "What is the capital of France?"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
```

---

## **Push to Hugging Face Hub (Optional)**

If you want to share your fine-tuned model, you can push it to the Hugging Face Hub.

```python
from huggingface_hub import notebook_login

# Log in to Hugging Face Hub
notebook_login()

# Push the model and tokenizer
model.push_to_hub("your-username/deepseek-v1.5-7b-finetuned")
tokenizer.push_to_hub("your-username/deepseek-v1.5-7b-finetuned")
```

---

## **Conclusion**

This guide demonstrates how to fine-tune the **DeepSeek-V1.5 7B** model for reasoning tasks using LoRA and Hugging Face libraries. By following these steps, you can adapt the model to your specific use case and save it for future inference.
