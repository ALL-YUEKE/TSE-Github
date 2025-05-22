import json
import os
import glob
import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import argparse

# Select a GPU with free memory between 10,000 MB and 50,000 MB
parser = argparse.ArgumentParser(description="Process language parameter")
parser.add_argument("--lang",type=str,required=True)
args = parser.parse_args()
def select_gpu(min_free_mem_mb=10000, max_free_mem_mb=50000):
    gpu_memory = [
        (
            i, 
            (torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)) // (1024 * 1024)  # Convert to MB
        ) 
        for i in range(torch.cuda.device_count())
    ]
    for gpu_id, free_mem_mb in gpu_memory:
        if min_free_mem_mb < free_mem_mb < max_free_mem_mb:
            return gpu_id
    raise RuntimeError(f"No GPU with free memory between {min_free_mem_mb} MB and {max_free_mem_mb} MB available.")


# Save Evaluation Results
def save_evaluation_results(eval_results, predictions, labels, test_df, train_df, label_map, output_dir):
    # Ensure predictions and labels are converted to lists
    predictions = predictions.tolist() if hasattr(predictions, "tolist") else predictions
    labels = labels.tolist() if hasattr(labels, "tolist") else labels

    # Save final accuracy
    final_accuracy = eval_results.get("eval_accuracy", None)
    results = {"accuracy": final_accuracy}

    # Save predictions and true labels
    results["predictions"] = predictions
    results["true_labels"] = labels

    # Add corresponding added_code and author to the predictions
    results["test_items"] = test_df.to_dict(orient="records")

    # Calculate per-label accuracy and counts
    per_label_metrics = {}
    for label_name, label_idx in label_map.items():
        label_count = len(train_df[train_df["emailname"] == label_name])  # Number of samples in the dataset for the label
        test_mask = [i for i, lbl in enumerate(labels) if lbl == label_idx]

        if len(test_mask) > 0:
            correct_predictions = sum(
                1 for i in test_mask if predictions[i] == label_idx
            )
            label_accuracy = correct_predictions / len(test_mask)
        else:
            label_accuracy = 2  # If the label is not present in the test set

        per_label_metrics[label_name] = {
            "accuracy": label_accuracy,
            "count": label_count
        }

    results["per_label_metrics"] = per_label_metrics

    # Save results to a JSON file
    output_path = os.path.join(output_dir, "evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Evaluation results saved to {output_path}")


# Custom Dataset Class
class CodeStylometryDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# Metric Function
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


# Load and preprocess data from JSON
def load_and_preprocess_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Convert JSON to DataFrame
    records = []
    for item in data:
        added_code = "\n".join(item.get("added_code", []))  # Combine added_code lines into a single string
        records.append({
            "added_code": added_code,
            "emailname": item["emailname"]
        })

    df = pd.DataFrame(records)

    # Remove classes with fewer than 2 samples
    class_counts = df["emailname"].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    df = df[df["emailname"].isin(valid_classes)]

    return df


# Tokenize and Encode Dataset
def preprocess_data(df, tokenizer, label_map):
    encodings = tokenizer(df["added_code"].tolist(), truncation=True, padding=True, max_length=512)
    labels = [label_map[label] for label in df["emailname"]]
    return encodings, labels


# Shuffle Dataset
def shuffle_dataframe(df):
    return df.sample(frac=1).reset_index(drop=True)


# Main Fine-Tuning Code
def fine_tune(json_path, model_ckpt="microsoft/graphcodebert-base", output_dir=f"./graphcodebert_{args.lang}"):
    print(f"Processing file: {json_path}")

    # Load and preprocess data
    df = load_and_preprocess_data(json_path)

    # Check if the dataset is empty after filtering
    if df.empty:
        print(f"Skipping file {json_path}: No valid classes with at least 2 samples.")
        return

    # Shuffle data
    df = shuffle_dataframe(df)

    # Create label map
    unique_authors = df["emailname"].unique()
    label_map = {author: idx for idx, author in enumerate(unique_authors)}

    # Split data into train and test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["emailname"])

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    # Preprocess data
    train_encodings, train_labels = preprocess_data(train_df, tokenizer, label_map)
    test_encodings, test_labels = preprocess_data(test_df, tokenizer, label_map)

    # Create datasets
    train_dataset = CodeStylometryDataset(train_encodings, train_labels)
    test_dataset = CodeStylometryDataset(test_encodings, test_labels)

    # Select GPU
    gpu_id = select_gpu()
    torch.cuda.set_device(gpu_id)
    print(f"Using GPU: {gpu_id}")

    # Load model
    num_labels = len(unique_authors)
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).to("cuda")

    # Training Arguments
    model_output_dir = os.path.join(output_dir, os.path.basename(json_path).replace(".json", ""))
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
        logging_steps=10
    )

    # Trainer
    early_stopping_callback = EarlyStoppingCallback(threshold_loss=0.03, patience=5)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback]
    )

    # Train and Evaluate
    trainer.train()
    eval_results = trainer.evaluate()

    # Generate predictions
    predictions = trainer.predict(test_dataset).predictions
    predicted_labels = np.argmax(predictions, axis=1)

    # Save evaluation results
    save_evaluation_results(
        eval_results,
        predicted_labels,
        test_labels,
        test_df,
        train_df,
        label_map,
        model_output_dir
    )

    print(f"Finished processing file: {json_path}")


# Run Fine-Tuning for All JSON Files
if __name__ == "__main__":
    json_dir = f"language/combined_{args.lang}"  # Directory containing the JSON files
    json_files = glob.glob(os.path.join(json_dir, "*.json"))  # Get all JSON files in the directory

    if not json_files:
        print(f"No JSON files found in directory: {json_dir}")
        exit()

    for json_file in json_files:  # Process only the first 2 files for testing
        try:
            fine_tune(json_file)
        except Exception as e:
            print(f"Error processing file {json_file}: {e}")

