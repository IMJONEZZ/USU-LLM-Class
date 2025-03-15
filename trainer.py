from zenml import step
import sys
import torch
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

sys.stdout.reconfigure(encoding="utf-8")


@step
def SFT_train(train_dataset):
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "6GB", "cpu": "12GB"},
    )

    model.gradient_checkpointing_enable()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,  # Accumulate gradients to reduce memory
        save_steps=10_000,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=500,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset.dataset,
        data_collator=lambda data: {
            "input_ids": torch.stack([f["input_ids"] for f in data]),
            "attention_mask": torch.stack([f["attention_mask"] for f in data]),
            "labels": torch.stack([f["input_ids"] for f in data]),
        },
    )

    trainer.train()

    model.save_pretrained("./assn_6_llama")
    tokenizer.save_pretrained("./assn_6_llama")

    print("Model training complete and saved to './assn_6_llama'")
