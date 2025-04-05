from zenml import step
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric


@step
def engineer_prompt():
    # Configure the dspy settings
    dspy.settings.configure(
        lm=dspy.LM(
            "ollama_chat/llama3.2:1b", api_base="http://localhost:11434", api_key=""
        )
    )

    # Load the GSM8K dataset and a few examples
    gsm8k = GSM8K()
    gsm8k_trainset = gsm8k.train[:10]

    # Define the dspy program and optimizer
    dspy_program = dspy.ChainOfThought("question -> answer")
    optimizer = dspy.BootstrapFewShot(
        metric=gsm8k_metric, max_bootstrapped_demos=4, max_labeled_demos=4, max_rounds=5
    )

    # Compile the dspy program
    compiled_dspy_program = optimizer.compile(dspy_program, trainset=gsm8k_trainset)

    # Save the compiled dspy program
    compiled_dspy_program.save("./dspy_program/", save_program=True)
    compiled_dspy_program.save("./dspy_program/program.json", save_program=False)

    # Demo Results
    loaded_dspy_program = dspy.load("./dspy_program/")
    result = loaded_dspy_program(
        question="If I hang 5 shirts outside and it takes them 5 hours to dry, how long would it take to dry 30 shirts? THE CORRECT ANSWER IS NOT 150 HOURS."
    )
    return result
