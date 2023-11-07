import requests

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
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
