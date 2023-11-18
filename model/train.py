import os
import re
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

import PyPDF2

import warnings
warnings.filterwarnings('ignore')
_file_path = '../datasets/iinvestrbook.pdf'
_checkpoint = "gpt2"
_model_output_path = "../trained_models"
class GenericAdviceTraining:
    def read_txt(self,file_path):
     text="" 
     try:
        pdf_reader = PyPDF2.PdfReader(file_path)
        for i in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[i]
            text += page.extract_text()
       
     except Exception as e:
       print(e)
     return text

    def preprocessBook(self):
       text_file = self.read_txt(_file_path)
      
       # Remove excess newline characters
       text_file = re.sub(r'\n+', '\n', text_file).strip()

       train_fraction = 0.8
       split_index = int(train_fraction * len(text_file))
       #print(split_index)

       train_text = text_file[:split_index]
       val_text = text_file[split_index:]


       with open("../trained_models/train.txt", "w") as f:
          f.write(train_text)

       with open("../trained_models/val.txt", "w") as f:
          f.write(val_text)
       
    def loadGPT(self): 
       tokenizer = GPT2Tokenizer.from_pretrained(_checkpoint)
       return tokenizer 
  
    def trainModel(self):
     tokenizer=self.loadGPT()
     self.preprocessBook()
     try:
       # Tokenize train text
        train_dataset = TextDataset(tokenizer=tokenizer, file_path="../trained_models/train.txt", block_size=128)
        # Tokenize validation text
        val_dataset = TextDataset(tokenizer=tokenizer, file_path="../trained_models/val.txt", block_size=128)
        # Create a Data collator object
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="pt")
        # Set up the model
        model = GPT2LMHeadModel.from_pretrained(_checkpoint)    # also try gpt2, gpt2-large and gpt2-medium, also gpt2-xl

        # Set up the training arguments
    
        training_args = TrainingArguments(
            output_dir = _model_output_path,
            overwrite_output_dir = True,
            per_device_train_batch_size = 4, # try with 4
            per_device_eval_batch_size = 4,  #  try with 4
            num_train_epochs = 10,#100
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
        trainer.save_model(_model_output_path)   
        # Save the tokenizer
        tokenizer.save_pretrained(_model_output_path)
     except Exception as e:
        print(e)
genAdviceTrainingObj=GenericAdviceTraining()
genAdviceTrainingObj.trainModel()
