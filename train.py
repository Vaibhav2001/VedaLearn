import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
# loading tokenizer from the saved model path
save_path = "tokens"
tokenizer = GPT2Tokenizer.from_pretrained(save_path, additional_special_tokens=["<s>","<pad>","</s>","<unk>","<mask>"],pad_token='<pad>', max_len=512)
# creating the configurations from which the model can be made
config = GPT2Config(
  vocab_size=tokenizer.vocab_size,
)
# creating the model
model = GPT2LMHeadModel(config)

from transformers import TextDataset

dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="final.txt",
    block_size=128,
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="checkpoints",
    overwrite_output_dir=True,
    num_train_epochs=15,
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model("model")

