from data_loader import load_data
from tokenizer import encode_text
from dataset import split_data
from train import train_model

### Sample Test Data ###
SAMPLE_TEXT = "Hello world. ZenML is great!"
SAMPLE_TOKENS = [[101, 7592, 2088, 102], [101, 10924, 2638, 2003, 2307, 102]]
SAMPLE_SPLIT = {
    "train": SAMPLE_TOKENS[:1],
    "val": SAMPLE_TOKENS[:1],
    "test": SAMPLE_TOKENS[:1],
}



### 2️⃣ Tokenization ###
def test_encode_text():
    result = encode_text(SAMPLE_TEXT)
    assert isinstance(result, list)
    assert all(isinstance(seq, list) for seq in result)
    assert len(result) > 0


### 3️⃣ Data Splitting ###
def test_split_data():
    result = split_data(SAMPLE_TOKENS)
    assert isinstance(result, dict)
    assert all(key in result for key in ["train", "val", "test"])
    assert sum(len(v) for v in result.values()) == len(SAMPLE_TOKENS)


### 4️⃣ Model Training ###
def test_train_model():
    result = train_model(SAMPLE_SPLIT, batch_size=1, epochs=1)
    assert isinstance(result, dict)
    assert "train_loss" in result
