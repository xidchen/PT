# !pip install accelerate bitsandbytes datasets peft transformers trl

import datasets
import peft
import torch
import transformers
import trl


def create_and_prepare_model():
    compute_dtype = getattr(torch, "float16")
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path="tiiuae/falcon-7b",
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map={"": 0},
    )
    model = peft.prepare_model_for_kbit_training(model)
    peft_config = peft.LoraConfig(
        r=64,
        target_modules=["query_key_value"],
        task_type="CAUSAL_LM",
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
    )
    model = peft.get_peft_model(model, peft_config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="tiiuae/falcon-7b",
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, peft_config, tokenizer


# noinspection PyTypeChecker
def run_falcon_qlora():
    training_arguments = transformers.TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        fp16=True,
        max_grad_norm=0.3,
        # max_steps=1000,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
    )

    model, peft_config, tokenizer = create_and_prepare_model()
    model.config.use_cache = False
    dataset = datasets.load_dataset(
        path="timdettmers/openassistant-guanaco",
        split="train"
    )

    trainer = trl.SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        # packing=True,
    )

    trainer.train()


if __name__ == "__main__":
    run_falcon_qlora()
