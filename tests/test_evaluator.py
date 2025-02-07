from nltk.stem import WordNetLemmatizer
import nltk
from rouge_score import rouge_scorer

# Ensure necessary NLTK resources are downloaded
nltk.download("wordnet")

# Initialize lemmatizer and ROUGE scorer
lemmatizer = WordNetLemmatizer()
rouge_scorer = rouge_scorer


# Function to lemmatize a sentence
def lemmatize_sentence(sentence):
    return " ".join([lemmatizer.lemmatize(word) for word in sentence.split()])


# Test cases
test_cases = [
    (
        "The river rushes through the valley.",
        "The river flows swiftly through the valley.",
    ),
    ("She enjoys reading science fiction.", "She likes to read sci-fi books."),
    ("He is a talented musician.", "He is a skilled performer."),
    ("The wind is blowing strongly.", "The wind blows hard."),
    ("They are planning a trip to Europe.", "They are traveling to Europe soon."),
]

# Run tests
for i, (prediction, reference) in enumerate(test_cases):
    # Apply lemmatization
    pred_lem = lemmatize_sentence(prediction)
    ref_lem = lemmatize_sentence(reference)

    # Compute ROUGE scores
    scores = rouge_scorer.RougeScorer(pred_lem, ref_lem)

    # Display results
    print(f"Test {i + 1}:")
    print("Original Prediction: ", prediction)
    print("Original Reference:  ", reference)
    print("Lemmatized Prediction:", pred_lem)
    print("Lemmatized Reference: ", ref_lem)
    print("ROUGE Scores:", scores)
    print("-" * 50)
