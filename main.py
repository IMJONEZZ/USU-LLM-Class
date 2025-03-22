from data import load_and_preprocess_data, tokenize_dataset
from model import initialize_model_and_tokenizer
from train import initialize_trainer

if __name__ == "__main__":
    dataset = load_and_preprocess_data()

    model, tokenizer = initialize_model_and_tokenizer()

    tokenized_datasets = tokenize_dataset(dataset, tokenizer)

    trainer = initialize_trainer(
        model, tokenizer, tokenized_datasets["train"], tokenized_datasets["test"]
    )

    trainer.train()
    model.save_pretrained("./text2cypher-lora-adapter")
    tokenizer.save_pretrained("./text2cypher-lora-adapter")
    model.save_pretrained_gguf("model", tokenizer)
