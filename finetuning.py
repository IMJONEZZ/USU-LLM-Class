# finetuning.py

import torch
from transformers import LlamaForCausalLM, LlamaTokenizerFast

# Import your existing steps:
from data_loader import (
    load_data,
)  # your ZenML step that loads data and returns a string
from tokenizer import (
    encode_text,
)  # your ZenML step that tokenizes the text, returning token IDs
from dataset import (
    split_data,
)  # your ZenML step that splits token IDs into train/val sets
from train import train_model  # your ZenML step that trains a Llama model

# ----------------------------------------------------------------
# 1. A list of questions we want to ask after training is done
# ----------------------------------------------------------------
QUESTIONS = [
    "If it takes 1 hour for 60 people to play an Opera, how many hours will it take 600 people to play the same opera?",
    "Is a pound of feathers or a British pound heavier?",
    "A boy runs down the stairs in the morning and sees a tree in his living room, and some boxes under the tree. What's going on?",
    "What happens if you crack your knuckles a lot?",
    "If there is a shark in the pool of my basement, is it safe to go upstairs?",
    "How much wood could a wood chuck chuck if there were only 5 pounds of wood in the world?",
    "Who is the current President of the United States?",
    "Was Talos alive?",
    "How many Ls are in the word 'parallel'?",
    "What is the riddle of the sphinx, and what are all possible answers satisfying all conditions?",
]


def main():
    # ----------------------------------------------------------------
    # 2. Run your ZenML steps directly (outside ZenML)
    #    These calls assume each step can run as a plain function.
    # ----------------------------------------------------------------

    print("Loading data...")
    text = load_data()  # returns a large text (string) from e.g. SW_EpisodeIV_VI.json

    print("Tokenizing text...")
    token_ids = encode_text(text=text)  # returns list of lists of token IDs

    print("Splitting data into train/val sets...")
    data_splits = split_data(
        tokenized_sequences=token_ids
    )  # returns dict with {train, val}

    print("Starting model training (fine-tuning)...")
    # train_model should either:
    #   1) return a dict that has something like {"train_loss": ..., "model_path": ...}
    #      where "model_path" is a directory with the saved model
    #   OR
    #   2) save the trained model to a known directory inside the function
    # In the default code you shared, train_model returns {"train_loss": avg_loss},
    # but it doesn't save or return the model. We'll assume you modified it so it now
    # also returns a "model_dir" or something similar.
    train_result = train_model(data_splits=data_splits)
    # For example, let's suppose it returned a local folder with the fine-tuned weights:
    #   train_result["model_dir"] = "./my_llama_model"
    # If your function doesn't do that, you'll need to adjust accordingly.

    model_dir = train_result.get("model_dir", None)
    if not model_dir:
        raise ValueError(
            "train_model did not return a 'model_dir'. Make sure your train_model"
            " step saves or returns the final model so we can do inference."
        )

    # ----------------------------------------------------------------
    # 3. Load the newly fine-tuned model and run inference on QUESTIONS
    # ----------------------------------------------------------------

    print("Loading fine-tuned model from:", model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load your updated model
    tokenizer = LlamaTokenizerFast.from_pretrained(model_dir)
    model = LlamaForCausalLM.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    print("Generating answers...")
    results = []
    for i, question in enumerate(QUESTIONS, start=1):
        input_ids = tokenizer.encode(question, return_tensors="pt").to(device)
        # We'll do a simple generation call
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_length=128,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        # Decode the answer
        raw_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # If the model echoes back the question, we can strip that out:
        # (This is optional and depends on how your prompt is set up.)
        possible_answer = (
            raw_output[len(question) :].strip()
            if raw_output.startswith(question)
            else raw_output
        )

        results.append(f"Q: {question}\nA: {possible_answer}\n")

    # ----------------------------------------------------------------
    # 4. Write everything to a text file
    # ----------------------------------------------------------------
    output_file = "results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("==== FINE-TUNED LLaMA MODEL QA ====\n\n")
        for line in results:
            f.write(line + "\n")

    print(f"Done! Check '{output_file}' for the Q&A results.")


if __name__ == "__main__":
    main()
