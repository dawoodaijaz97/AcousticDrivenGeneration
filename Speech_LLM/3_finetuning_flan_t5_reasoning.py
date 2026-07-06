"""
{'learning_rate': 1e-04, 'num_train_epochs': 12, 'per_device_train_batch_size': 16}

"""

import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer,T5Tokenizer, DataCollatorForSeq2Seq, Trainer, Seq2SeqTrainingArguments, T5ForConditionalGeneration, TrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback
from rouge_score import rouge_scorer
import wandb
import nltk
import numpy as np
import evaluate
from pytorch_lightning import seed_everything

wandb.login(key='3512381bd8ea2874b219d463de226e59e9ef989b')


class T5_QuestionAnswering:
    def __init__(self, model_name="google/flan-t5-small"):
        self.model_name = 'google/'+model_name
        # self.prefix = "generate the report for: "
        self.prefix = "Consider the numbers of each item in the following text and generate a report: "
        self.tokenizer =  AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.datacollator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)

    def prepare_dataset_for_t5(self, train_df, val_df, test_df):
        train_df = train_df.astype(str)
        train_data_dict = {'Instructions': train_df['Instructions'], 'Report': train_df['Report']}
        train_dataset = Dataset.from_dict(train_data_dict)

        val_df = val_df.astype(str)
        val_data_dict = {'Instructions': val_df['Instructions'], 'Report': val_df['Report']}
        val_dataset = Dataset.from_dict(val_data_dict)

        test_df = test_df.astype(str)
        test_data_dict = {'Instructions': test_df['Instructions'], 'Report': test_df['Report']}
        test_dataset = Dataset.from_dict(test_data_dict)

        return train_dataset, val_dataset, test_dataset

    def preprocess_function(self, examples):
        inputs = [self.prefix + doc for doc in examples['Instructions']]
        model_inputs = self.tokenizer(inputs, max_length=128, truncation=True)

        labels = self.tokenizer(examples["Report"], max_length=512, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def tokenize_dataset(self, dataset):
        tokenized_dataset = dataset.map(self.preprocess_function, batched=True)
        return tokenized_dataset

    def train_model(self, tokenized_dataset,model_name="model",model_type="small"):
        
        L_RATE = 3e-4
        BATCH_SIZE = 8
        PER_DEVICE_EVAL_BATCH = 4
        WEIGHT_DECAY = 0.01
        SAVE_TOTAL_LIM = 3
        NUM_EPOCHS = 3
        
        training_args = Seq2SeqTrainingArguments(
            output_dir="./models/"+model_name,
            overwrite_output_dir=True,
            report_to="wandb",
            logging_dir="./logs/"+model_name,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=L_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
            weight_decay=WEIGHT_DECAY,
            save_total_limit=SAVE_TOTAL_LIM,
            num_train_epochs=NUM_EPOCHS,
            predict_with_generate=True,
            load_best_model_at_end = True,
            push_to_hub=False
            )      
        print("Training Arguments:", training_args)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=self.datacollator,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)] 
        )

        trainer.train()
        return trainer

    def save_model(self, trainer,model_name,model_type):
        self.model.save_pretrained("models/"+model_name)
        self.tokenizer.save_pretrained("models/"+model_name)

    def load_saved_model(self, model_dir):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        tokenizer =  AutoTokenizer.from_pretrained(model_dir)
        return model, tokenizer

    def generate_response(self, prompt, model, tokenizer):
        inputs = [self.prefix + prompt]
        inputs = tokenizer(inputs, max_length=128, truncation=True, return_tensors="pt")
        
        output = model.generate(**inputs,
                                num_beams=3,
                                max_length=512,
                                no_repeat_ngram_size=2,
                                top_k=50,
                                top_p=0.95,
                                temperature=0.7,
                                pad_token_id=tokenizer.eos_token_id,
                                do_sample=True)
        
        response = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

        return response



def main(model_type="small"):
    
    sample_size = '10k'
    model_name = sample_size+"_flan-t5_"+model_type+'reasoning'
        
    wandb.init(
        project="IS_2025_mFDA_LLM",
        name=model_name
    )
    
    t5_qa = T5_QuestionAnswering(model_name='flan-t5-'+model_type)
    
    #-
    train_df = pd.read_csv('./data/Reports/'+sample_size+'_samples/train_split.csv')
    val_df = pd.read_csv('./data/Reports/'+sample_size+'_samples/val_split.csv')
    test_df = pd.read_csv('./data/Reports/'+sample_size+'_samples/test_split.csv')
    
    
    #-
    train_dataset, val_dataset, test_dataset = t5_qa.prepare_dataset_for_t5(train_df, val_df, test_df)
    tokenized_train_dataset = t5_qa.tokenize_dataset(train_dataset)
    tokenized_val_dataset = t5_qa.tokenize_dataset(val_dataset)
    tokenized_test_dataset = t5_qa.tokenize_dataset(test_dataset)
    tokenized_datasets = {"train": tokenized_train_dataset, "validation": tokenized_val_dataset, "test": tokenized_test_dataset}
    print("tokenized_datasets:", tokenized_datasets)

    trainer = t5_qa.train_model(tokenized_datasets,model_name,model_type)
    t5_qa.save_model(trainer,model_name,model_type)

if __name__ == "__main__":
    
    model_type = "large"

    main(model_type = model_type)
    wandb.finish()






