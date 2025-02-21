from unsloth import FastLanguageModel
import huggingface_hub
from config import MODEL_NAME, quant_config, lora_config


def initialize_model_and_tokenizer():
    huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = False
    huggingface_hub.constants.DEFAULT_ETAG_TIMEOUT = 30

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        quantization_config=quant_config,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config.r,
        target_modules=lora_config.target_modules,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    return model, tokenizer
