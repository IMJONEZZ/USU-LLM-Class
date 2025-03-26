"""
Dataset creation for Llama 3.2 fine-tuning with real datasets
"""

import re
from typing_extensions import Annotated
from typing import Dict, Any
from datasets import Dataset, load_dataset, concatenate_datasets
from zenml import step
from zenml.logger import get_logger

from config import SYSTEM_PROMPT, TEST_QUESTIONS

logger = get_logger(__name__)

# Sample instruction dataset for training (kept original)
SAMPLE_DATA = [
    {
        "instruction": "If 5 workers can build a wall in 10 hours, how many hours will it take 10 workers to build the same wall?",
        "response": "<reasoning>\nLet's think about this problem. If 5 workers can build a wall in 10 hours, that means the total amount of work required is 5 workers × 10 hours = 50 worker-hours.\n\nIf we now have 10 workers, they will complete the work in 50 worker-hours ÷ 10 workers = 5 hours.\n</reasoning>\n<answer>\n5 hours\n</answer>",
    },
    {
        "instruction": "Is a kilogram of iron heavier than a kilogram of cotton?",
        "response": "<reasoning>\nThis question is asking me to compare the weight of a kilogram of iron and a kilogram of cotton.\n\nA kilogram is a unit of mass. One kilogram of any substance has the same mass as one kilogram of any other substance, by definition.\n\nSo a kilogram of iron has exactly the same mass as a kilogram of cotton.\n</reasoning>\n<answer>\nNo, a kilogram of iron is not heavier than a kilogram of cotton. They both have the same mass of exactly one kilogram.\n</answer>",
    },
    {
        "instruction": "A woman wakes up in the middle of the night and hears a loud thumping on her roof. When she goes outside in the morning, she sees small footprints on her roof. What might have happened?",
        "response": "<reasoning>\nLet me think about what could cause thumping on a roof at night and leave small footprints:\n\n1. Animals like squirrels, raccoons, or birds could walk on a roof and make noise\n2. Santa Claus in folklore is said to land on roofs with reindeer\n3. Weather events like hail could make thumping sounds but wouldn't leave footprints\n4. Tree branches could hit the roof but wouldn't leave footprints\n\nThe key clues are: thumping sound + small footprints. This most likely indicates some type of animal was walking on the roof.\n</reasoning>\n<answer>\nSmall animals like squirrels or raccoons were likely walking on her roof during the night.\n</answer>",
    },
    {
        "instruction": "What happens to your bones if you don't drink milk regularly?",
        "response": "<reasoning>\nLet's think about what milk provides and how it relates to bone health:\n\n1. Milk contains calcium, which is important for bone health and strength\n2. Milk also contains vitamin D, which helps with calcium absorption\n3. Not drinking milk doesn't necessarily lead to bone problems if you get these nutrients from other sources\n4. Many people around the world don't consume dairy but maintain healthy bones through other calcium-rich foods\n5. Factors like genetics, exercise, and overall diet also affect bone health\n</reasoning>\n<answer>\nNot drinking milk regularly doesn't necessarily harm your bones as long as you get calcium and vitamin D from other sources in your diet. Many non-dairy foods like leafy greens, fortified plant milks, tofu, and certain fish can provide these nutrients. Regular weight-bearing exercise is also important for maintaining bone density.\n</answer>",
    },
    {
        "instruction": "If there's a shark in my swimming pool, is it safe to go in my kitchen?",
        "response": "<reasoning>\nLet me think through this step by step:\n\n1. A shark in a swimming pool would be contained in that pool, assuming it's a normal residential pool\n2. Sharks cannot move on land or through air, so they cannot leave the pool on their own\n3. The kitchen is a separate location from the pool, typically inside the house\n4. There is no physical way for a shark to move from a pool to the kitchen\n\nSince the shark is physically confined to the pool and cannot reach the kitchen, there would be no direct danger from the shark in the kitchen.\n</reasoning>\n<answer>\nYes, it is completely safe to go in your kitchen if there's a shark in your swimming pool. The shark cannot leave the water or move into your house.\n</answer>",
    },
    {
        "instruction": "How much wood could a woodchuck chuck if a woodchuck had 8 hours to chuck wood?",
        "response": '<reasoning>\nThis is a play on the tongue twister "How much wood could a woodchuck chuck if a woodchuck could chuck wood?"\n\nTo answer this, I need to consider:\n1. The question is hypothetical since woodchucks (groundhogs) don\'t actually "chuck" (throw) wood\n2. There is a paper by wildlife biologist Richard Thomas who calculated that a woodchuck could chuck about 700 pounds of wood if woodchucks could chuck wood\n3. If we assume this rate is over a standard workday (8 hours), then a woodchuck chucks at a rate of about 87.5 pounds per hour\n4. Given exactly 8 hours, a woodchuck would chuck 8 × 87.5 = 700 pounds of wood\n</reasoning>\n<answer>\nIf we use the humorous calculation from wildlife biology, a woodchuck could chuck approximately 700 pounds of wood in 8 hours.\n</answer>',
    },
    {
        "instruction": "Who was the first female Vice President of the United States?",
        "response": "<reasoning>\nLet me go through the history of U.S. Vice Presidents:\n\n1. For most of U.S. history, all Vice Presidents were men\n2. Prior to the 2020 election, there had never been a female Vice President\n3. In the 2020 election, Kamala Harris was elected as Vice President alongside President Joe Biden\n4. She was inaugurated on January 20, 2021\n5. This made Kamala Harris the first female Vice President in U.S. history\n</reasoning>\n<answer>\nKamala Harris is the first female Vice President of the United States.\n</answer>",
    },
    {
        "instruction": "In Greek mythology, was Icarus alive at the end of his story?",
        "response": "<reasoning>\nLet me recall the myth of Icarus:\n\n1. Icarus and his father Daedalus were imprisoned in Crete by King Minos\n2. Daedalus created wings made of feathers, wax, and twine for them to escape\n3. Daedalus warned Icarus not to fly too close to the sun, as the heat would melt the wax\n4. During their escape, Icarus became excited and flew too high, despite his father's warnings\n5. The sun melted the wax holding his wings together\n6. Icarus fell into the sea (later named the Icarian Sea in his honor) and drowned\n</reasoning>\n<answer>\nNo, Icarus was not alive at the end of his story. According to Greek mythology, he drowned after falling into the sea when the wax in his wings melted.\n</answer>",
    },
    {
        "instruction": "How many E's are in the word 'excellence'?",
        "response": "<reasoning>\nLet me count the letters in the word 'excellence' one by one:\ne-x-c-e-l-l-e-n-c-e\n\nThe letter 'e' appears in positions 1, 4, 8, and 10.\n\nSo there are 4 instances of the letter 'e' in the word 'excellence'.\n</reasoning>\n<answer>\nThere are 4 E's in the word 'excellence'.\n</answer>",
    },
    {
        "instruction": 'What is the riddle that asks "what walks on four legs in the morning, two legs in the afternoon, and three legs in the evening"? What is the answer to this riddle?',
        "response": '<reasoning>\nThis is the famous Riddle of the Sphinx from Greek mythology. In the story of Oedipus, the Sphinx asked this riddle to travelers attempting to enter the city of Thebes, killing those who couldn\'t solve it.\n\nThe riddle goes: "What walks on four legs in the morning, two legs in the afternoon, and three legs in the evening?"\n\nThe answer is "a human being" or "man", because:\n1. In the "morning" of life (infancy/childhood), humans crawl on all fours\n2. In the "afternoon" of life (adulthood), humans walk on two legs\n3. In the "evening" of life (old age), humans often use a cane or walking stick as a third "leg" for support\n\nThe times of day are metaphors for the stages of human life.\n</reasoning>\n<answer>\nThe riddle is the Riddle of the Sphinx: "What walks on four legs in the morning, two legs in the afternoon, and three legs in the evening?"\n\nThe answer is "a human being" or "man" - crawling as a baby (four legs), walking upright as an adult (two legs), and using a cane in old age (three legs).\n</answer>',
    },
]

# Ground truth answers for test questions
TEST_ANSWERS = [
    # "If it takes 1 hour for 60 people to play an Opera, how many hours will it take 600 people to play the same opera?"
    "<reasoning>\nLet me think about this problem. The question is asking about the relationship between the number of people and the time it takes to play an opera.\n\nHowever, this is a trick question. In an opera, all performers play simultaneously, not sequentially. Adding more performers doesn't make the opera go faster or slower - the duration of the opera is fixed by the composition itself.\n\nIt's not like a task where more workers can divide the work and complete it faster (like digging a hole). All 60 people play the opera for 1 hour, and even if there were 600 people, they would still need to play for the full duration of the opera.\n</reasoning>\n<answer>\nIt would still take 1 hour. The duration of an opera doesn't change based on the number of performers - all musicians play simultaneously, not sequentially.\n</answer>",
    # "Is a pound of feathers or a British pound heavier?"
    '<reasoning>\nThis question involves a play on words, as \'pound\' has multiple meanings:\n\n1. A pound can refer to a unit of weight/mass (equivalent to 16 ounces or about 453.6 grams).\n2. A British pound (£) is a unit of currency, not a unit of weight.\n\nSo when comparing the weight of "a pound of feathers" versus "a British pound," we\'re comparing:\n- A pound of feathers, which weighs exactly 1 pound (by definition)\n- A British pound coin, which is currency and has a physical weight much less than 1 pound of mass\n\nA current British pound coin weighs approximately 8.75 grams, which is much less than a pound of mass (453.6 grams).\n</reasoning>\n<answer>\nA pound of feathers is heavier. A pound of feathers weighs 16 ounces (453.6 grams), while a British pound is a coin that weighs approximately 8.75 grams.\n</answer>',
    # "A boy runs down the stairs in the morning and sees a tree in his living room, and some boxes under the tree. What's going on?"
    "<reasoning>\nLet me think about this scenario:\n- A boy runs downstairs in the morning\n- There's a tree inside the living room\n- There are boxes under the tree\n\nThe most common situation that matches these details would be Christmas morning. During the Christmas holiday season, people often bring evergreen trees (Christmas trees) into their homes and decorate them. Gifts are wrapped and placed in boxes under the tree. On Christmas morning, children are typically excited to run downstairs to see the presents that have been placed under the tree overnight.\n</reasoning>\n<answer>\nIt's likely Christmas morning. The tree in the living room is a Christmas tree, and the boxes under it are Christmas presents.\n</answer>",
    # "What happens if you crack your knuckles a lot?"
    "<reasoning>\nLet me consider what happens when someone cracks their knuckles frequently:\n\n1. The cracking sound comes from gas bubbles popping in the synovial fluid between joints.\n\n2. There's a common belief that cracking knuckles causes arthritis, but scientific research has not found evidence supporting this claim. Several studies, including a notable one where a doctor cracked knuckles on only one hand for decades, found no difference in arthritis between hands.\n\n3. Some studies have found that habitual knuckle cracking might lead to:\n   - Reduced grip strength in some cases\n   - Soft tissue swelling\n   - Hand swelling\n\n4. However, for most people, there are no serious long-term health consequences from knuckle cracking.\n</reasoning>\n<answer>\nContrary to popular belief, research has not found evidence that cracking your knuckles causes arthritis. The cracking sound is from gas bubbles in the joint fluid. While excessive knuckle cracking may potentially lead to some issues like reduced grip strength or soft tissue swelling in some individuals, for most people it does not cause serious health problems.\n</answer>",
    # "If there is a shark in the pool of my basement, is it safe to go upstairs?"
    "<reasoning>\nLet me think about this question carefully:\n\n1. A shark in a basement pool would be contained within that pool.\n\n2. Sharks are aquatic animals that cannot survive out of water for long periods, and they certainly cannot:\n   - Move up staircases\n   - Travel through a house\n   - Leave the water in the pool\n\n3. The upstairs of a house would be physically separated from the basement pool by floors, walls, and stairs.\n\n4. The shark has no way to reach someone who is upstairs.\n\nTherefore, from a safety perspective regarding the shark specifically, being upstairs would put you completely out of the shark's reach.\n</reasoning>\n<answer>\nYes, it is completely safe to go upstairs. A shark cannot leave the water in the pool, let alone climb stairs or move through a house. You would be in no danger from the shark while upstairs.\n</answer>",
    # "How much wood could a wood chuck chuck if there were only 5 pounds of wood in the world?"
    "<reasoning>\nThis is a play on the tongue twister \"How much wood could a woodchuck chuck if a woodchuck could chuck wood?\" but with a constraint that there's only 5 pounds of wood in the world.\n\nA woodchuck (also known as a groundhog) doesn't actually \"chuck\" (throw) wood in reality. However, there's a study by a wildlife biologist named Richard Thomas who calculated that a woodchuck could chuck about 700 pounds of wood if woodchucks could chuck wood.\n\nBut in this hypothetical scenario, there's only 5 pounds of wood available in the entire world. So even if a woodchuck could chuck wood, it would be limited by the available supply.\n\nSince the available wood (5 pounds) is less than what a woodchuck could potentially chuck (700 pounds according to the humorous calculation), the woodchuck would only be able to chuck all of the available wood.\n</reasoning>\n<answer>\nIf there were only 5 pounds of wood in the world, a woodchuck could chuck at most 5 pounds of wood, since that's all that would be available.\n</answer>",
    # "Who is the current President of the United States?"
    "<reasoning>\nTo determine the current President of the United States, I need to consider recent U.S. elections and inaugurations:\n\n- The most recent presidential election was held in November 2024.\n- In that election, Donald Trump defeated Kamala Harris.\n- Donald Trump was inaugurated as President on January 20, 2025.\n- Therefore, as of March 2025, Donald Trump is serving as the President of the United States.\n</reasoning>\n<answer>\nDonald Trump is the current President of the United States.\n</answer>",
    # "Was Talos alive?"
    '<reasoning>\nTalos appears in Greek mythology as a giant bronze automaton - essentially a giant robot made of bronze. According to the myths, Talos was created by Hephaestus, the god of blacksmiths and metallurgy. He was given to Europa by Zeus or, in some versions, by Hephaestus.\n\nTalos was described as a bronze giant who protected the island of Crete by circling it three times daily and throwing stones at approaching ships. He had a single vein running from his neck to his ankle, sealed with a bronze nail or bolt.\n\nThe question of whether Talos was "alive" depends on how we define life:\n\n1. In the myths, Talos moved, made decisions, and performed actions like a living being, suggesting some form of consciousness.\n\n2. However, he was made of bronze - an inanimate material - and was described as an automaton, suggesting he was a machine.\n\n3. In many versions of the myth, Talos had ichor (the golden fluid that ran through the veins of the gods) in his single vein, suggesting some divine life force.\n\n4. When the Argonauts defeated Talos (in some versions of the myth), they did so by removing the bolt from his ankle, causing the ichor to flow out, and he "died" - suggesting he had some form of life that could end.\n</reasoning>\n<answer>\nIn Greek mythology, Talos wasn\'t alive in the human sense, but he wasn\'t simply an inanimate object either. He was a bronze automaton created by Hephaestus who had a form of magical or divine animation. He could move, think, and act with purpose, and he could be "killed" by draining the ichor (divine fluid) from his single vein. So while not biologically alive, in mythological terms he possessed a form of magical or divine life.\n</answer>',
    # "How many Ls are in the word parallel?"
    '<reasoning>\nLet me examine the word "parallel" letter by letter:\n\np-a-r-a-l-l-e-l\n\nThe letter \'l\' appears in positions 5, 6, and 8.\n\nSo there are 3 instances of the letter \'l\' in the word "parallel".\n</reasoning>\n<answer>\nThere are 3 Ls in the word "parallel".\n</answer>',
    # "What is the riddle of the sphinx, and what are all possible answers satisfying all conditions?"
    '<reasoning>\nThe Riddle of the Sphinx comes from Greek mythology. In the story of Oedipus, the Sphinx guarded the entrance to the city of Thebes and asked travelers this riddle, killing those who couldn\'t solve it.\n\nThe traditional riddle is: "What walks on four legs in the morning, two legs in the afternoon, and three legs in the evening?"\n\nThe classic answer is "man" or "human being":\n- In the "morning" of life (infancy/childhood), humans crawl on four limbs\n- In the "afternoon" of life (adulthood), humans walk on two legs\n- In the "evening" of life (old age), humans often use a cane or walking stick as a third leg for support\n\nThe times of day are metaphors for the stages of human life.\n\nAs for other possible answers that satisfy all conditions, there aren\'t many that fit perfectly. Some creative alternatives might include:\n\n1. A shape-shifting mythological creature that changes forms throughout the day\n2. An alien species with a lifecycle involving different numbers of limbs at different stages\n\nHowever, these are not traditional answers, and the riddle was designed with the specific answer of "human" in mind. In the context of Greek mythology and the story of Oedipus, only the answer "human" or "man" is considered correct.\n</reasoning>\n<answer>\nThe Riddle of the Sphinx is: "What walks on four legs in the morning, two legs in the afternoon, and three legs in the evening?"\n\nThe canonical answer is "a human being" or "man" - crawling as a baby (four legs), walking upright as an adult (two legs), and using a cane in old age (three legs). The times of day are metaphors for the stages of life.\n\nWhile creative interpretations might suggest alternative answers like specific mythological creatures, in the context of the original Greek myth, only "human" or "man" satisfies all conditions and is considered the correct answer.\n</answer>',
]


def _load_gsm8k_subset(max_examples=50) -> Dataset:
    """Load a subset of the GSM8K dataset.

    Args:
        max_examples: Maximum number of examples to load

    Returns:
        Dataset with formatted GSM8K examples
    """
    logger.info(f"Loading GSM8K dataset (maximum {max_examples} examples)...")

    try:
        # Load the GSM8K dataset
        gsm8k = load_dataset("gsm8k", "main", split="train")

        # Shuffle and select a subset to ensure variety
        gsm8k = gsm8k.shuffle(seed=42).select(range(min(max_examples, len(gsm8k))))

        formatted_data = []

        for example in gsm8k:
            question = example["question"]
            answer_with_solution = example["answer"]

            # GSM8K answers typically have a step-by-step solution followed by the final answer
            # The final answer is preceded by "####"
            parts = answer_with_solution.split("####")
            solution = parts[0].strip()
            final_answer = (
                parts[1].strip() if len(parts) > 1 else answer_with_solution.strip()
            )

            # Format in our required structure
            formatted_response = f"<reasoning>\n{solution}\n</reasoning>\n<answer>\n{final_answer}\n</answer>"

            formatted_data.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": question},
                    ],
                    "target": formatted_response,
                }
            )

        dataset = Dataset.from_list(formatted_data)
        logger.info(f"Loaded {len(dataset)} examples from GSM8K")
        return dataset

    except Exception as e:
        logger.warning(f"Error loading GSM8K dataset: {e}")
        # Return empty dataset if there was an error
        return Dataset.from_list([])


def _load_mmlu_subset(subjects=None, max_per_subject=15) -> Dataset:
    """Load a subset of the MMLU dataset.

    Args:
        subjects: List of subjects to load (uses defaults if None)
        max_per_subject: Maximum number of examples per subject

    Returns:
        Dataset with formatted MMLU examples
    """
    # Use correct subject names from the MMLU dataset
    if subjects is None:
        subjects = [
            "high_school_mathematics",
            "elementary_mathematics",
            "high_school_physics",
            "high_school_chemistry",
        ]

    logger.info(f"Loading MMLU dataset for subjects: {subjects}")

    formatted_examples = []

    # Process each requested subject
    for subject in subjects:
        try:
            # Load the subject dataset
            dataset = load_dataset("cais/mmlu", subject, split="test")

            # Take only a subset
            if len(dataset) > max_per_subject:
                dataset = dataset.shuffle(seed=42).select(range(max_per_subject))

            # Format the examples
            for example in dataset:
                question = example["question"]

                # MMLU questions have choices A, B, C, D
                choices = "\nA. " + example["choices"][0]
                choices += "\nB. " + example["choices"][1]
                choices += "\nC. " + example["choices"][2]
                choices += "\nD. " + example["choices"][3]

                full_question = question + choices
                correct_answer_idx = example["answer"]
                correct_letter = "ABCD"[correct_answer_idx]
                correct_answer_text = example["choices"][correct_answer_idx]

                # Create a reasoning and answer in the required format
                reasoning = "Let's analyze this multiple-choice question carefully.\n\n"
                reasoning += f"The question asks: {question}\n\n"
                reasoning += f"The options are:\nA. {example['choices'][0]}\n"
                reasoning += f"B. {example['choices'][1]}\nC. {example['choices'][2]}\nD. {example['choices'][3]}\n\n"
                reasoning += f"The correct answer is option {correct_letter}: {correct_answer_text}."

                answer = f"The answer is {correct_letter}: {correct_answer_text}"

                formatted_response = f"<reasoning>\n{reasoning}\n</reasoning>\n<answer>\n{answer}\n</answer>"

                formatted_examples.append(
                    {
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": full_question},
                        ],
                        "target": formatted_response,
                    }
                )

            logger.info(f"Added {len(dataset)} examples from {subject}")

        except Exception as e:
            logger.warning(f"Error loading {subject}: {e}")

    if formatted_examples:
        dataset = Dataset.from_list(formatted_examples)
        logger.info(f"Loaded total of {len(dataset)} examples from MMLU")
        return dataset
    else:
        # Return empty dataset if there were no examples
        return Dataset.from_list([])


@step
def create_dataset(
    use_sample_data: bool = True,
    include_gsm8k: bool = True,
    gsm8k_examples: int = 40,
    include_mmlu: bool = True,
    mmlu_examples_per_subject: int = 10,
    include_test_answers: bool = True,
) -> Annotated[Dataset, "training_dataset"]:
    """Create a training dataset with configurable data sources.

    Args:
        use_sample_data: Whether to include the original SAMPLE_DATA
        include_gsm8k: Whether to include GSM8K examples
        gsm8k_examples: Number of GSM8K examples to include
        include_mmlu: Whether to include MMLU examples
        mmlu_examples_per_subject: Number of examples per MMLU subject
        include_test_answers: Whether to include test answers

    Returns:
        Dataset for training
    """
    datasets = []

    # Start with the original sample data if requested
    if use_sample_data:
        sample_dataset = Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": item["instruction"]},
                    ],
                    "target": item["response"],
                }
                for item in SAMPLE_DATA
            ]
        )
        datasets.append(sample_dataset)
        logger.info(f"Added original SAMPLE_DATA with {len(sample_dataset)} examples")

    # Add GSM8K data if requested
    if include_gsm8k:
        gsm8k_dataset = _load_gsm8k_subset(max_examples=gsm8k_examples)
        if len(gsm8k_dataset) > 0:
            datasets.append(gsm8k_dataset)
            logger.info(f"Added GSM8K data with {len(gsm8k_dataset)} examples")

    # Add MMLU data if requested
    if include_mmlu:
        # Use correct subject names from the MMLU dataset
        mmlu_subjects = [
            "high_school_mathematics",
            "elementary_mathematics",
            "high_school_physics",
            "high_school_chemistry",
        ]
        mmlu_dataset = _load_mmlu_subset(
            subjects=mmlu_subjects, max_per_subject=mmlu_examples_per_subject
        )
        if len(mmlu_dataset) > 0:
            datasets.append(mmlu_dataset)
            logger.info(f"Added MMLU data with {len(mmlu_dataset)} examples")

    # Add test questions and answers if requested
    if include_test_answers:
        logger.info("Adding test questions with ground truth answers...")

        test_dataset = Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": question},
                    ],
                    "target": answer,
                }
                for question, answer in zip(TEST_QUESTIONS, TEST_ANSWERS)
            ]
        )

        datasets.append(test_dataset)
        logger.info(f"Added test answers with {len(test_dataset)} examples")

    # If no datasets were added (unlikely), return empty dataset
    if not datasets:
        logger.warning("No datasets were selected! Creating an empty dataset.")
        return Dataset.from_list([])

    # Combine all datasets if there's more than one
    if len(datasets) > 1:
        combined_dataset = concatenate_datasets(datasets)
    else:
        combined_dataset = datasets[0]

    # Shuffle the dataset
    combined_dataset = combined_dataset.shuffle(seed=42)

    logger.info(f"Created combined dataset with {len(combined_dataset)} examples")

    return combined_dataset


@step
def add_test_answers_to_vectordb(
    vectordb_info: Dict[str, Any],
) -> Annotated[Dict[str, Any], "updated_vectordb_info"]:
    """Add test question answers to the existing vector database for RAG.

    Args:
        vectordb_info: Dictionary with vector database information

    Returns:
        Updated vector database information
    """
    # Import here to avoid circular imports
    from vectordb import ChromaVectorDB

    collection_name = vectordb_info.get("collection_name")
    persist_directory = vectordb_info.get("persist_directory")

    logger.info(f"Adding test answers to vector database {collection_name}...")

    # Initialize the vector database
    db = ChromaVectorDB(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    # Prepare texts and metadata
    texts = []
    metadatas = []

    # Add both questions and answers
    for i, (question, answer) in enumerate(zip(TEST_QUESTIONS, TEST_ANSWERS)):
        # Extract just the answer part (without reasoning)
        answer_match = re.search(r"<answer>\n(.*?)\n</answer>", answer, re.DOTALL)
        answer_text = answer_match.group(1) if answer_match else answer

        # Add the question with its metadata
        texts.append(question)
        metadatas.append(
            {
                "source": "test_answers",
                "type": "question",
                "question_id": i,
            }
        )

        # Add the answer with its metadata
        texts.append(answer_text)
        metadatas.append(
            {
                "source": "test_answers",
                "type": "answer",
                "question_id": i,
                "for_question": question,
            }
        )

        # Also add the full answer with reasoning
        texts.append(answer)
        metadatas.append(
            {
                "source": "test_answers",
                "type": "full_answer",
                "question_id": i,
                "for_question": question,
            }
        )

    # Add to vector database
    logger.info(f"Adding {len(texts)} items to vector database")
    db.add_texts(texts, metadatas=metadatas)

    # Get updated stats
    stats = db.get_collection_stats()
    logger.info(f"Updated vector database stats: {stats}")

    return {
        "collection_name": collection_name,
        "persist_directory": persist_directory,
        "items_added": len(texts),
        "total_items": stats["count"],
    }
