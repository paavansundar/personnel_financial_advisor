
import os
import re
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

import PyPDF2

import warnings
warnings.filterwarnings('ignore')
_file_path = './datasets/iinvestrbook.pdf'
_checkpoint = "gpt2"
_model_output_path = "./trained_models"
class GenericAdvice:   
    def generate_response(self,model, tokenizer, prompt, max_length=100):
        print("prompt is ",prompt)
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
        response=tokenizer.decode(output[0], skip_special_tokens=True)
        #print (response)
        return response

    def chat(self,prompt):
        my_model = GPT2LMHeadModel.from_pretrained(_model_output_path)
        my_tokenizer = GPT2Tokenizer.from_pretrained(_model_output_path)
        response = self.generate_response(my_model, my_tokenizer, prompt)
        return response
       
