import torch
import numpy as np
import evaluate
from transformers import BertTokenizer


def flatten_list(lst):
    """Recursively flatten a nested list into a single list."""
    flat = []
    for item in lst:
        if isinstance(item, list):
            flat.extend(flatten_list(item))
        else:
            flat.append(item)
    return flat


def compute_metrics(
    eval_preds,
    tokenizer=BertTokenizer.from_pretrained("bert-base-uncased"),
    bleu_metric=evaluate.load("bleu"),
):
    predictions, labels = eval_preds
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().tolist()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().tolist()
    if isinstance(predictions, (list, np.ndarray)) and np.ndim(predictions) > 2:
        predictions = np.argmax(predictions, axis=-1)

    def flatten_ids(seq):
        return [x[0] if isinstance(x, list) else x for x in seq]

    predictions = [flatten_ids(seq) for seq in predictions]
    labels = [flatten_ids(label) for label in labels]
    labels = [
        [(lab if lab != -100 else tokenizer.pad_token_id) for lab in label]
        for label in labels
    ]
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [ref.strip() for ref in decoded_labels]
    decoded_preds = [pred if len(pred) > 0 else " " for pred in decoded_preds]

    references = [[ref] for ref in decoded_labels]
    try:
        result = bleu_metric.compute(
            predictions=decoded_preds, references=references, smooth=True
        )
    except ZeroDivisionError:
        result = {"bleu": 0.0}

    pred_lens = [len(pred.split()) for pred in decoded_preds]
    result["gen_len"] = np.mean(pred_lens)
    result = {
        k: (round(v * 100, 4) if isinstance(v, (int, float)) else v)
        for k, v in result.items()
    }
    return result
