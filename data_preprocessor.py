from typing import Dict, List, Tuple, Any, Optional
from typing_extensions import Annotated
import re
from datasets import Dataset, load_dataset
from zenml import step, log_artifact_metadata
from zenml.logger import get_logger

logger = get_logger(__name__)

# System prompt for our model
SYSTEM_PROMPT = """You are a helpful, harmless, and precise assistant. When responding, first think through the problem step-by-step, then provide your final answer.

Respond in the following format:
<reasoning>
[Your step-by-step thinking about the problem]
</reasoning>
<answer>
[Your final, concise answer]
</answer>
"""

# Sample instruction dataset for training
SAMPLE_DATA = [
    {
        "instruction": "If 5 workers can build a wall in 10 hours, how many hours will it take 10 workers to build the same wall?",
        "response": "<reasoning>\nLet's think about this problem. If 5 workers can build a wall in 10 hours, that means the total amount of work required is 5 workers × 10 hours = 50 worker-hours.\n\nIf we now have 10 workers, they will complete the work in 50 worker-hours ÷ 10 workers = 5 hours.\n</reasoning>\n<answer>\n5 hours\n</answer>"
    },
    {
        "instruction": "Is a kilogram of iron heavier than a kilogram of cotton?",
        "response": "<reasoning>\nThis question is asking me to compare the weight of a kilogram of iron and a kilogram of cotton.\n\nA kilogram is a unit of mass. One kilogram of any substance has the same mass as one kilogram of any other substance, by definition.\n\nSo a kilogram of iron has exactly the same mass as a kilogram of cotton.\n</reasoning>\n<answer>\nNo, a kilogram of iron is not heavier than a kilogram of cotton. They both have the same mass of exactly one kilogram.\n</answer>"
    },
    {
        "instruction": "A woman wakes up in the middle of the night and hears a loud thumping on her roof. When she goes outside in the morning, she sees small footprints on her roof. What might have happened?",
        "response": "<reasoning>\nLet me think about what could cause thumping on a roof at night and leave small footprints:\n\n1. Animals like squirrels, raccoons, or birds could walk on a roof and make noise\n2. Santa Claus in folklore is said to land on roofs with reindeer\n3. Weather events like hail could make thumping sounds but wouldn't leave footprints\n4. Tree branches could hit the roof but wouldn't leave footprints\n\nThe key clues are: thumping sound + small footprints. This most likely indicates some type of animal was walking on the roof.\n</reasoning>\n<answer>\nSmall animals like squirrels or raccoons were likely walking on her roof during the night.\n</answer>"
    },
    {
        "instruction": "What happens to your bones if you don't drink milk regularly?",
        "response": "<reasoning>\nLet's think about what milk provides and how it relates to bone health:\n\n1. Milk contains calcium, which is important for bone health and strength\n2. Milk also contains vitamin D, which helps with calcium absorption\n3. Not drinking milk doesn't necessarily lead to bone problems if you get these nutrients from other sources\n4. Many people around the world don't consume dairy but maintain healthy bones through other calcium-rich foods\n5. Factors like genetics, exercise, and overall diet also affect bone health\n</reasoning>\n<answer>\nNot drinking milk regularly doesn't necessarily harm your bones as long as you get calcium and vitamin D from other sources in your diet. Many non-dairy foods like leafy greens, fortified plant milks, tofu, and certain fish can provide these nutrients. Regular weight-bearing exercise is also important for maintaining bone density.\n</answer>"
    },
    {
        "instruction": "If there's a shark in my swimming pool, is it safe to go in my kitchen?",
        "response": "<reasoning>\nLet me think through this step by step:\n\n1. A shark in a swimming pool would be contained in that pool, assuming it's a normal residential pool\n2. Sharks cannot move on land or through air, so they cannot leave the pool on their own\n3. The kitchen is a separate location from the pool, typically inside the house\n4. There is no physical way for a shark to move from a pool to the kitchen\n\nSince the shark is physically confined to the pool and cannot reach the kitchen, there would be no direct danger from the shark in the kitchen.\n</reasoning>\n<answer>\nYes, it is completely safe to go in your kitchen if there's a shark in your swimming pool. The shark cannot leave the water or move into your house.\n</answer>"
    },
    {
        "instruction": "How much wood could a woodchuck chuck if a woodchuck had 8 hours to chuck wood?",
        "response": "<reasoning>\nThis is a play on the tongue twister \"How much wood could a woodchuck chuck if a woodchuck could chuck wood?\"\n\nTo answer this, I need to consider:\n1. The question is hypothetical since woodchucks (groundhogs) don't actually \"chuck\" (throw) wood\n2. There is a paper by wildlife biologist Richard Thomas who calculated that a woodchuck could chuck about 700 pounds of wood if woodchucks could chuck wood\n3. If we assume this rate is over a standard workday (8 hours), then a woodchuck chucks at a rate of about 87.5 pounds per hour\n4. Given exactly 8 hours, a woodchuck would chuck 8 × 87.5 = 700 pounds of wood\n</reasoning>\n<answer>\nIf we use the humorous calculation from wildlife biology, a woodchuck could chuck approximately 700 pounds of wood in 8 hours.\n</answer>"
    },
    {
        "instruction": "Who was the first female Vice President of the United States?",
        "response": "<reasoning>\nLet me go through the history of U.S. Vice Presidents:\n\n1. For most of U.S. history, all Vice Presidents were men\n2. Prior to the 2020 election, there had never been a female Vice President\n3. In the 2020 election, Kamala Harris was elected as Vice President alongside President Joe Biden\n4. She was inaugurated on January 20, 2021\n5. This made Kamala Harris the first female Vice President in U.S. history\n</reasoning>\n<answer>\nKamala Harris is the first female Vice President of the United States.\n</answer>"
    },
    {
        "instruction": "In Greek mythology, was Icarus alive at the end of his story?",
        "response": "<reasoning>\nLet me recall the myth of Icarus:\n\n1. Icarus and his father Daedalus were imprisoned in Crete by King Minos\n2. Daedalus created wings made of feathers, wax, and twine for them to escape\n3. Daedalus warned Icarus not to fly too close to the sun, as the heat would melt the wax\n4. During their escape, Icarus became excited and flew too high, despite his father's warnings\n5. The sun melted the wax holding his wings together\n6. Icarus fell into the sea (later named the Icarian Sea in his honor) and drowned\n</reasoning>\n<answer>\nNo, Icarus was not alive at the end of his story. According to Greek mythology, he drowned after falling into the sea when the wax in his wings melted.\n</answer>"
    },
    {
        "instruction": "How many E's are in the word 'excellence'?",
        "response": "<reasoning>\nLet me count the letters in the word 'excellence' one by one:\ne-x-c-e-l-l-e-n-c-e\n\nThe letter 'e' appears in positions 1, 4, 8, and 10.\n\nSo there are 4 instances of the letter 'e' in the word 'excellence'.\n</reasoning>\n<answer>\nThere are 4 E's in the word 'excellence'.\n</answer>"
    },
    {
        "instruction": "What is the riddle that asks \"what walks on four legs in the morning, two legs in the afternoon, and three legs in the evening\"? What is the answer to this riddle?",
        "response": "<reasoning>\nThis is the famous Riddle of the Sphinx from Greek mythology. In the story of Oedipus, the Sphinx asked this riddle to travelers attempting to enter the city of Thebes, killing those who couldn't solve it.\n\nThe riddle goes: \"What walks on four legs in the morning, two legs in the afternoon, and three legs in the evening?\"\n\nThe answer is \"a human being\" or \"man\", because:\n1. In the \"morning\" of life (infancy/childhood), humans crawl on all fours\n2. In the \"afternoon\" of life (adulthood), humans walk on two legs\n3. In the \"evening\" of life (old age), humans often use a cane or walking stick as a third \"leg\" for support\n\nThe times of day are metaphors for the stages of human life.\n</reasoning>\n<answer>\nThe riddle is the Riddle of the Sphinx: \"What walks on four legs in the morning, two legs in the afternoon, and three legs in the evening?\"\n\nThe answer is \"a human being\" or \"man\" - crawling as a baby (four legs), walking upright as an adult (two legs), and using a cane in old age (three legs).\n</answer>"
    }
]

def extract_structured_answer(text):
    """Extract reasoning and answer sections from a structured response."""
    # Default values
    reasoning = ""
    answer = ""

    # Extract reasoning section
    reasoning_pattern = r"<reasoning>(.*?)</reasoning>"
    reasoning_match = re.search(reasoning_pattern, text, re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    # Extract answer section
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()

    # If no structured format found, use the entire text as the answer
    if not reasoning and not answer:
        answer = text.strip()

    return reasoning, answer

@step
def create_dataset() -> Annotated[Dataset, "training_dataset"]:
    """Create a dataset for instruction tuning.
    
    This step loads a dataset of math problems and formats them in a way
    suitable for fine-tuning the LLaMA model. It adds reasoning-based
    formatting to teach the model structured reasoning patterns.

    Returns:
        Dataset: A formatted dataset ready for fine-tuning
    """
    # Load high-quality reasoning dataset (GSM8K)
    logger.info("Loading GSM8K dataset...")
    dataset = load_dataset('openai/gsm8k', 'main')['train']

    # Simple formatting function 
    def format_example(example):
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example['question']}
            ],
            "target": f"<reasoning>\n{example['answer']}\n</reasoning>\n<answer>\n{example['answer'].split('####')[-1].strip()}\n</answer>"
        }

    # Apply formatting
    logger.info("Formatting GSM8K examples...")
    formatted_dataset = dataset.map(format_example)

    # Add original sample data
    logger.info("Adding specific examples with structured reasoning...")
    original_data = [
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["instruction"]}
            ],
            "target": item["response"]
        } for item in SAMPLE_DATA
    ]

    # Combine datasets
    full_dataset = Dataset.from_list(list(formatted_dataset) + original_data)
    
    # Log dataset statistics
    dataset_stats = {
        "total_examples": len(full_dataset),
        "gsm8k_examples": len(formatted_dataset),
        "custom_examples": len(original_data)
    }
    log_artifact_metadata(dataset_stats)

    logger.info(f"Created dataset with {len(full_dataset)} examples")

    return full_dataset


@step
def test_questions() -> Annotated[List[str], "test_questions"]:
    """Generate the list of test questions for the assignment.
    
    This step returns the specific test questions that will be used to evaluate
    the model after fine-tuning.
    
    Returns:
        List[str]: A list of test questions
    """
    TEST_QUESTIONS = [
        "If it takes 1 hour for 60 people to play an Opera, how many hours will it take 600 people to play the same opera?",
        "Is a pound of feathers or a British pound heavier?",
        "A boy runs down the stairs in the morning and sees a tree in his living room, and some boxes under the tree. What's going on?",
        "What happens if you crack your knuckles a lot?",
        "If there is a shark in the pool of my basement, is it safe to go upstairs?",
        "How much wood could a wood chuck chuck if there were only 5 pounds of wood in the world?",
        "Who is the current President of the United States?",
        "Was Talos alive?",
        "How many Ls are in the word parallel?",
        "What is the riddle of the sphinx, and what are all possible answers satisfying all conditions?"
    ]
    
    logger.info(f"Prepared {len(TEST_QUESTIONS)} test questions for evaluation")
    
    return TEST_QUESTIONS