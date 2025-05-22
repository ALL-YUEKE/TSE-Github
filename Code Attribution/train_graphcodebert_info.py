import json
import os
import glob
import torch
import pandas as pd
import numpy as np
import warnings
import random
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import argparse
warnings.filterwarnings("ignore")

# Set seed for reproducibility
parser = argparse.ArgumentParser(description="Process language parameter")
parser.add_argument("--lang",type=str,required=True)
args = parser.parse_args()
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# GPU selection
def select_gpu(min_free_mem_mb=10000, max_free_mem_mb=50000):
    try:
        gpu_memory = [
            (
                i,
                (torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)) // (1024 * 1024)
            )
            for i in range(torch.cuda.device_count())
        ]
        for gpu_id, free_mem_mb in gpu_memory:
            if min_free_mem_mb < free_mem_mb < max_free_mem_mb:
                return gpu_id
    except Exception as e:
        print(f"GPU selection error: {e}")
    return 0  # Fallback to CPU if no suitable GPU found

# Save evaluation results
def save_evaluation_results(eval_results, predictions, labels, test_df, output_dir, label_encoder):
    try:
        # Ensure predictions and labels are numpy arrays
        predictions = np.array(predictions)
        labels = np.array(labels)

        # Convert numeric labels back to original author names
        predicted_authors = label_encoder.inverse_transform(predictions)
        true_authors = label_encoder.inverse_transform(labels)

        # Compute detailed classification report
        report = classification_report(labels, predictions, output_dict=True)

        # Create label-to-author mapping
        label_to_author = {int(label): author for label, author in enumerate(label_encoder.classes_)}

        results = {
            "accuracy": eval_results.get("eval_accuracy", None),
            "predictions": predicted_authors.tolist(),
            "true_labels": true_authors.tolist(),
            "classification_report": report,
            "label_to_author": label_to_author,
            "test_items": test_df.to_dict(orient="records")
        }

        output_path = os.path.join(output_dir, "evaluation_results_info.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Evaluation results saved to {output_path}")
    except Exception as e:
        print(f"Error saving evaluation results: {e}")

# Custom Dataset Class
class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, df, code_tokenizer, text_tokenizer, label_encoder):
        # Tokenize code
        self.code_encodings = code_tokenizer(
            df["added_code"].tolist(),
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )

        # Tokenize text messages
        self.text_encodings = text_tokenizer(
            df["message"].tolist(),
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )

        # Create filename mapping
        unique_filenames = df["filename"].unique()
        filename_map = {name: idx for idx, name in enumerate(unique_filenames)}
        self.filename_ids = torch.tensor(
            df["filename"].map(filename_map).values,
            dtype=torch.long
        )

        # Encode labels
        self.labels = torch.tensor(
            label_encoder.transform(df["emailname"]),
            dtype=torch.long
        )

    def __getitem__(self, idx):
        return {
            "code_input_ids": self.code_encodings["input_ids"][idx],
            "code_attention_mask": self.code_encodings["attention_mask"][idx],
            "text_input_ids": self.text_encodings["input_ids"][idx],
            "text_attention_mask": self.text_encodings["attention_mask"][idx],
            "filename_ids": self.filename_ids[idx],
            "labels": self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)

# Compute metrics for evaluation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

# Early Stopping Callback
class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, threshold_loss=0.1, patience=3):
        self.threshold_loss = threshold_loss
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def on_evaluate(self, args, state, control, **kwargs):
        eval_loss = kwargs["metrics"].get("eval_loss")
        if eval_loss is not None:
            if eval_loss < self.threshold_loss:
                print(f"Stopping early: eval_loss={eval_loss} is below the threshold {self.threshold_loss}")
                control.should_training_stop = True
            elif eval_loss < self.best_loss:
                self.best_loss = eval_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"Stopping early: eval_loss did not improve for {self.patience} evaluations.")
                    control.should_training_stop = True

# Multi-Modal Model
class MultiModalModel(torch.nn.Module):
    def __init__(self, code_model_ckpt, text_model_ckpt, num_filenames, num_labels):
        super(MultiModalModel, self).__init__()
        self.num_labels = num_labels
        self.code_encoder = AutoModel.from_pretrained(code_model_ckpt)
        self.text_encoder = AutoModel.from_pretrained(text_model_ckpt)
        self.filename_embedding = torch.nn.Embedding(num_filenames, 32)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(
                self.code_encoder.config.hidden_size + self.text_encoder.config.hidden_size + 32, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, num_labels)
        )

    def forward(self, code_input_ids, code_attention_mask, text_input_ids, text_attention_mask, filename_ids, labels=None):
        # Code encoder
        code_output = self.code_encoder(
            input_ids=code_input_ids,
            attention_mask=code_attention_mask
        )

        # Text encoder
        text_output = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        )

        # For code_output and text_output, get the pooled representation
        # If 'pooler_output' exists, use it; else, use the [CLS] token's hidden state
        if hasattr(code_output, 'pooler_output') and code_output.pooler_output is not None:
            code_pooled_output = code_output.pooler_output
        else:
            code_pooled_output = code_output.last_hidden_state[:, 0, :]  # [CLS] token

        if hasattr(text_output, 'pooler_output') and text_output.pooler_output is not None:
            text_pooled_output = text_output.pooler_output
        else:
            text_pooled_output = text_output.last_hidden_state[:, 0, :]  # [CLS] token

        # Filename embedding
        filename_output = self.filename_embedding(filename_ids)

        # Concatenate all inputs
        combined = torch.cat((code_pooled_output, text_pooled_output, filename_output), dim=1)

        # Classification logits
        logits = self.fc(combined)

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)

# Collate function to handle batching
def collate_fn(batch):
    # Check if batch is valid and has at least one item
    if not batch:
        raise ValueError("Empty batch received")
    
    # Prepare batch tensors
    return {
        "code_input_ids": torch.stack([item["code_input_ids"] for item in batch]),
        "code_attention_mask": torch.stack([item["code_attention_mask"] for item in batch]),
        "text_input_ids": torch.stack([item["text_input_ids"] for item in batch]),
        "text_attention_mask": torch.stack([item["text_attention_mask"] for item in batch]),
        "filename_ids": torch.stack([item["filename_ids"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch])
    }

# Main fine-tuning function
def fine_tune(json_path, 
              code_model_ckpt="microsoft/graphcodebert-base", 
              text_model_ckpt="distilbert-base-uncased", 
              output_dir=f"./graphcodebert_{args.lang}_info",
              seed=42):
    print(f"Processing file: {json_path}")

    # Set seeds for reproducibility
    set_seed(seed)
    
    # Load data
    try:
        with open(json_path, "r", encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file {json_path}: {e}")
        return
    
    # Prepare DataFrame
    records = []
    for item in data:
        added_code = "\n".join(item.get("added_code", [])) if item.get("added_code") else ""
        message = item.get("message", "")
        filename = item.get("filename", "")
        emailname = item.get("emailname", "")
        
        if added_code or message:
            records.append({
                "added_code": added_code,
                "message": message,
                "filename": filename,
                "emailname": emailname
            })
    
    df = pd.DataFrame(records)
    
    # Filter out authors with insufficient samples
    class_counts = df["emailname"].value_counts()
    print("Class counts:", class_counts)
    valid_classes = class_counts[class_counts >= 3].index
    df = df[df["emailname"].isin(valid_classes)]
    
    if df.empty:
        print(f"No valid classes with at least 2 samples in {json_path}")
        return
    
    # Shuffle data
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Create label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(df["emailname"])
    try:    
    # Split data
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["emailname"])
    except:
        print('data ineffient')
        return
    # Initialize tokenizers
    code_tokenizer = AutoTokenizer.from_pretrained(code_model_ckpt)
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_ckpt)
    
    # Create datasets
    train_dataset = MultiModalDataset(train_df, code_tokenizer, text_tokenizer, label_encoder)
    test_dataset = MultiModalDataset(test_df, code_tokenizer, text_tokenizer, label_encoder)
    
    # GPU selection
    gpu_id = select_gpu()
    torch.cuda.set_device(gpu_id)
    print(f"Using GPU: {gpu_id}")
    
    # Model configuration
    num_labels = len(label_encoder.classes_)
    num_filenames = len(df["filename"].unique())
    
    # Create output directory
    model_output_dir = os.path.join(output_dir, os.path.basename(json_path).replace(".json", ""))
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Initialize model
    model = MultiModalModel(code_model_ckpt, text_model_ckpt, num_filenames, num_labels).to("cuda")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=20,
        weight_decay=0.01,
        save_strategy="no",
        logging_dir=f"{model_output_dir}/logs",
        logging_steps=10,
        seed=seed  # Set seed for TrainingArguments
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=code_tokenizer,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(threshold_loss=0.1, patience=3)]
    )
    
    try:
        # Train the model
        trainer.train()
        
        # Evaluate the model
        eval_results = trainer.evaluate()
        predictions = trainer.predict(test_dataset).predictions
        predicted_labels = np.argmax(predictions, axis=1)
        
        # Save evaluation results
        save_evaluation_results(
            eval_results, 
            predicted_labels, 
            test_dataset.labels.numpy(), 
            test_df, 
            model_output_dir,
            label_encoder
        )
        
        print(f"Finished processing file: {json_path}")
    except Exception as e:
        print(f"Training error for {json_path}: {e}")

# Run for all JSON files
if __name__ == "__main__":
    # Set seeds for reproducibility
    seed = 42
    set_seed(seed)

    json_dir = f"language/combined_{args.lang}"

    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    if not json_files:
        print(f"No JSON files found in directory: {json_dir}")
        exit()
    
    for json_file in json_files:
        fine_tune(json_file, seed=seed)

