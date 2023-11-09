'''import requests

API_URL = "https://api-inference.huggingface.co/models/Tathagat/llama2-financial-advisor"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "Can you please let us know more details about your ",
})

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="jirin/Llama-2-13b-fingpt2")

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("jirin/Llama-2-13b-fingpt2")
model = AutoModelForCausalLM.from_pretrained("jirin/Llama-2-13b-fingpt2")

from transformers import AutoTokenizer, LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]'''

import os
import re
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

import warnings
warnings.filterwarnings('ignore')


def read_txt(file_path):
    with open(file_path, "r") as file:
        text = file.read()
    return text

file_path = '/content/35185-0.txt'
text_file = read_txt(file_path)

# Remove excess newline characters
text_file = re.sub(r'\n+', '\n', text_file).strip()

train_fraction = 0.8
split_index = int(train_fraction * len(text_file))

train_text = text_file[:split_index]
val_text = text_file[split_index:]


with open("train.txt", "w") as f:
    f.write(train_text)

with open("val.txt", "w") as f:
    f.write(val_text)
	
checkpoint = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)    # also try gpt2, gpt2-large and gpt2-medium, also gpt2-xl

sample_ids = tokenizer("Hello world")
sample_tokens = tokenizer.convert_ids_to_tokens(sample_ids['input_ids'])
# Generate original text back
tokenizer.convert_tokens_to_string(sample_tokens)

# Tokenize train text
train_dataset = TextDataset(tokenizer=tokenizer, file_path="train.txt", block_size=128)

# Tokenize validation text
val_dataset = TextDataset(tokenizer=tokenizer, file_path="val.txt", block_size=128)
# Create a Data collator object
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="pt")
# Set up the model
model = GPT2LMHeadModel.from_pretrained(checkpoint)    # also try gpt2, gpt2-large and gpt2-medium, also gpt2-xl

# Set up the training arguments

model_output_path = "/content/gpt_model"

training_args = TrainingArguments(
    output_dir = model_output_path,
    overwrite_output_dir = True,
    per_device_train_batch_size = 4, # try with 2
    per_device_eval_batch_size = 4,  #  try with 2
    num_train_epochs = 100,
    save_steps = 1_000,
    save_total_limit = 2,
    logging_dir = './logs',
    )
# Train the model
trainer = Trainer(
    model = model,
    args = training_args,
    data_collator = data_collator,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
)

trainer.train()

# Save the model
trainer.save_model(model_output_path)

# Save the tokenizer
tokenizer.save_pretrained(model_output_path)

def generate_response(model, tokenizer, prompt, max_length=100):

    input_ids = tokenizer.encode(prompt, return_tensors="pt")      # 'pt' for returning pytorch tensor

    # Create the attention mask and pad token id
    attention_mask = torch.ones_like(input_ids)
    pad_token_id = tokenizer.eos_token_id

    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        attention_mask=attention_mask,
        pad_token_id=pad_token_id
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)


my_model = GPT2LMHeadModel.from_pretrained(model_output_path)
my_tokenizer = GPT2Tokenizer.from_pretrained(model_output_path)


prompt = "What is teaching of Buddha?"  # Replace with your desired prompt
response = generate_response(my_model, my_tokenizer, prompt)
print("Generated response:", response)

# Testing with given prompt 2
prompt = "what is dharma ?"  # Replace with your desired prompt
response = generate_response(my_model, my_tokenizer, prompt, max_length=150)
print("Generated response:", response)

# Testing with given prompt 3

prompt = "how to live ?"  # Replace with your desired prompt
response = generate_response(my_model, my_tokenizer, prompt, max_length=150)
print("Generated response:", response)
