import torch
from metrics import compute_metrics
from transformers import (
    BertGenerationEncoder,
    BertGenerationDecoder,
    EncoderDecoderModel,
    BertTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset


def main():
    dataset = load_dataset("neo4j/text2cypher-2024v1")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["question"], truncation=True, max_length=128, padding="max_length"
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["cypher"],
                truncation=True,
                max_length=128,
                padding="max_length",
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=dataset["train"].column_names
    )

    tokenized_datasets.set_format(type="torch")
    subset_train = tokenized_datasets["train"].select(
        range(300)
    )  # make training on cpu manageable
    subset_val = tokenized_datasets["test"].select(
        range(50)
    )  # make training on cpu manageable

    encoder = BertGenerationEncoder.from_pretrained(
        "bert-base-uncased",
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
    )
    decoder = BertGenerationDecoder.from_pretrained(
        "bert-base-uncased",
        add_cross_attention=True,
        is_decoder=True,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
    )
    model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        evaluation_strategy="epoch",
        predict_with_generate=True,
        save_strategy="epoch",
        logging_steps=50,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=subset_train,
        eval_dataset=subset_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    results = trainer.evaluate()
    print("BLEU evaluation results:", results)

    torch.save(model, "model.pt")


if __name__ == "__main__":
    main()
