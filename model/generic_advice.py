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

