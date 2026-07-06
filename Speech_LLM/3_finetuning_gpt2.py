from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers.trainer_callback import EarlyStoppingCallback
import pandas as pd
from torch.utils.data import Dataset
import torch
from rouge_score import rouge_scorer
import wandb
# from pytorch_lightning import seed_everything

wandb.login(key='3512381bd8ea2874b219d463de226e59e9ef989b')

sample_size = '10k'
model_type = "small"
model_name = sample_size+"_gpt2_"+model_type
wandb.init(
    project="IS_2025_mFDA_LLM",
    name=model_name
)


class QADataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=128):
        self.examples = []

        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            instruction = row['Instructions']
            report = row['Report']
            text = f"Instruction: {instruction} Report: {report}"
            tokenized_text = tokenizer(text, truncation=True, max_length=block_size, padding="max_length")
            self.examples.append(tokenized_text)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return {key: torch.tensor(val) for key, val in self.examples[item].items()}

class Finetuning_gpt2:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.trainer = None

    def load_model(self):
        model_name = "gpt2"
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Add padding token to tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def prepare_datasets(self, path_to_train_dataset, path_to_val_dataset):
        self.train_dataset = QADataset(tokenizer=self.tokenizer, file_path=path_to_train_dataset)
        self.val_dataset = QADataset(tokenizer=self.tokenizer, file_path=path_to_val_dataset)

    def initialize_trainer(self):
        training_args = TrainingArguments(
            output_dir="./models/"+model_name,
            report_to="wandb",
            overwrite_output_dir=True,
            num_train_epochs=20,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=4,
            logging_dir="./logs/"+model_name,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_total_limit=2,
            learning_rate=3e-4,
            load_best_model_at_end=True,
            weight_decay=0.01,
        )
        print("Training arguments:", training_args)
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False),
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        )

    def train_model(self):
        self.trainer.train()

    def save_model(self):
        self.model.save_pretrained("models/"+model_name)
        self.tokenizer.save_pretrained("models/"+model_name)

    def load_saved_model(self, model_dir):
        model = GPT2LMHeadModel.from_pretrained(model_dir)
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set for the loaded tokenizer
        return model, tokenizer

    @staticmethod
    def generate_response(model, tokenizer, prompt, max_length=128):
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output = model.generate(
            input_ids,
            do_sample=True,
            max_length=max_length,
            num_beams=3,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response

def main():
    gpt2_qa = Finetuning_gpt2()
    gpt2_qa.load_model()
    gpt2_qa.prepare_datasets('./data/Reports/'+sample_size+'_samples/train_split.csv', './data/Reports/'+sample_size+'_samples/val_split.csv')
    gpt2_qa.initialize_trainer()
    gpt2_qa.train_model()
    gpt2_qa.save_model()

    
if __name__ == "__main__":
    main()
