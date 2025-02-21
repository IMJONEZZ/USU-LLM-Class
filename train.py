from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from config import training_kwargs


def get_training_arguments():
    return TrainingArguments(**training_kwargs)


def initialize_trainer(model, tokenizer, train_dataset, eval_dataset):
    return Trainer(
        model=model,
        args=get_training_arguments(**training_kwargs),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
