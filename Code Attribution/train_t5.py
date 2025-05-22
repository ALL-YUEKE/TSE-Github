import argparse
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    T5Config,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorWithPadding
)

warnings.filterwarnings("ignore")


##############################################################################
#                                ARG PARSING                                 #
##############################################################################

parser = argparse.ArgumentParser(description="Fine-tune CodeT5 for stylometry classification.")
parser.add_argument("--lang", type=str, required=True, help="Language code to process.")

args = parser.parse_args()


##############################################################################
#                             GPU SELECTION                                  #
##############################################################################
def select_gpu(min_free_mem_mb=10000, max_free_mem_mb=50000):
    """
    Returns the GPU index with the highest available memory in the specified range.
    If no GPU in range is found, defaults to CPU (returns 0).
    """
    try:
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            print("No GPU found; using CPU.")
            return 0  # CPU

        # Get (gpu_index, free_memory_in_MB) for each GPU
        gpu_memory = []
        for i in range(gpu_count):
            total_mem = torch.cuda.get_device_properties(i).total_memory
            used_mem = torch.cuda.memory_allocated(i)
            free_mb = (total_mem - used_mem) // (1024 * 1024)
            gpu_memory.append((i, free_mb))

        # Filter out GPUs outside the memory range
        candidates = [(gpu_id, free_mb)
                      for gpu_id, free_mb in gpu_memory
                      if min_free_mem_mb < free_mb < max_free_mem_mb]

        # If no candidate GPUs, fallback to CPU
        if not candidates:
            print("No GPU meets the memory requirement; using CPU.")
            return 0

        # Pick the GPU with the maximum free memory
        best_gpu = max(candidates, key=lambda x: x[1])[0]
        return best_gpu

    except Exception as e:
        print(f"GPU selection error: {e}")
        return 0  # Fallback to CPU on any error


##############################################################################
#                            DATASET & MODELING                              #
##############################################################################
class CodeStylometryDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for code stylometry classification tasks.
    """

    def __init__(self, encodings, labels):
        """
        Initialize the dataset.

        Args:
            encodings (dict): A dictionary of tokenized inputs (input_ids, attention_mask).
            labels (list[int]): List of integer labels corresponding to each sample.
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset by index.
        """
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": torch.tensor(self.labels[idx])
        }

    def __len__(self):
        """
        Return the length of the dataset.
        """
        return len(self.labels)


class T5ForCodeClassification(nn.Module):
    """
    A custom classification model built on top of T5’s encoder.
    This bypasses the T5 decoder and adds a linear classification head.
    """

    def __init__(self, model_name_or_path, num_labels):
        super().__init__()
        # Load T5 config
        from transformers import T5ForConditionalGeneration

        self.config = T5Config.from_pretrained(model_name_or_path)
        self.num_labels = num_labels

        # We only need T5’s encoder for classification; T5ForConditionalGeneration includes both.
        base_t5 = T5ForConditionalGeneration.from_pretrained(model_name_or_path)

        # We'll keep just the encoder portion
        self.encoder = base_t5.encoder

        # Classification head on top of the encoder’s pooled output
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.d_model, num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        # Pass inputs through T5 encoder
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        # encoder_outputs.last_hidden_state shape = (batch_size, seq_len, hidden_dim)
        last_hidden_state = encoder_outputs.last_hidden_state

        # Simple average pooling to get a single vector per sequence
        pooled_output = last_hidden_state.mean(dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}


##############################################################################
#                              METRICS & UTILS                               #
##############################################################################
def compute_metrics(eval_pred):
    """
    Compute the accuracy metric for evaluation.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}


class EarlyStoppingCallback(TrainerCallback):
    """
    A callback for early stopping based on evaluation loss.
    """

    def __init__(self, threshold_loss=0.1, patience=3):
        """
        Args:
            threshold_loss (float): Stop training if eval_loss < threshold_loss.
            patience (int): Number of evaluations without improvement before stopping.
        """
        self.threshold_loss = threshold_loss
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def on_evaluate(self, args, state, control, **kwargs):
        """
        Triggered at the end of each evaluation.
        """
        eval_loss = kwargs["metrics"].get("eval_loss")
        if eval_loss is not None:
            if eval_loss < self.threshold_loss:
                print(f"Stopping early: eval_loss={eval_loss:.4f} < {self.threshold_loss:.4f}")
                control.should_training_stop = True
            elif eval_loss < self.best_loss:
                self.best_loss = eval_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"Stopping early: eval_loss did not improve for {self.patience} evaluations.")
                    control.should_training_stop = True


def shuffle_dataframe(df):
    """
    Shuffle a DataFrame and reset its index.
    """
    return df.sample(frac=1).reset_index(drop=True)


##############################################################################
#                    EVALUATION RESULTS & SAVING TO JSON                     #
##############################################################################
def save_evaluation_results(eval_results, predicted_labels, test_labels, test_df, train_df, label_map, output_dir):
    """
    Saves evaluation details (accuracy, numeric->string label mapping, etc.) to a JSON file
    called 'evaluation_results_info.json'.
    """
    from sklearn.metrics import classification_report

    try:
        # Convert to numpy arrays
        predicted_labels = np.array(predicted_labels)
        test_labels = np.array(test_labels)

        # Invert label_map to get id->author
        id_to_author = {v: k for k, v in label_map.items()}

        # Convert numeric to string labels
        predicted_authors = [id_to_author[p] for p in predicted_labels]
        true_authors = [id_to_author[t] for t in test_labels]

        # Classification report
        report = classification_report(test_labels, predicted_labels, output_dict=True)

        # Build JSON results
        results = {
            "eval_accuracy": eval_results.get("eval_accuracy", None),
            "eval_loss": eval_results.get("eval_loss", None),
            "predicted_authors": predicted_authors,
            "true_authors": true_authors,
            "classification_report": report,
            "label_map": {str(k): v for k, v in id_to_author.items()},
            "test_samples": test_df.to_dict(orient="records"),
            "train_samples": train_df.to_dict(orient="records")
        }

        # Write to JSON file
        output_path = os.path.join(output_dir, "evaluation_results.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        print(f"Saved evaluation results to {output_path}")

    except Exception as e:
        print(f"Error saving evaluation results: {e}")


##############################################################################
#                                DATA LOADING                                #
##############################################################################
def load_and_preprocess_data(json_path):
    """
    Load data from a JSON file and filter out classes with fewer than 2 samples.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for item in data:
        added_code = "\n".join(item.get("added_code", []))
        records.append({
            "added_code": added_code,
            "emailname": item["emailname"]
        })

    df = pd.DataFrame(records)
    class_counts = df["emailname"].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    return df[df["emailname"].isin(valid_classes)]


def preprocess_data(df, tokenizer, label_map):
    """
    Tokenize text and convert labels to integer IDs.
    """
    encodings = tokenizer(
        df["added_code"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=512
    )
    labels = [label_map[label] for label in df["emailname"]]
    return encodings, labels


##############################################################################
#                             TRAINING LOGIC                                 #
##############################################################################
def fine_tune(json_path, model_ckpt="Salesforce/codet5p-220m", output_dir=f"./t5_{args.lang}"):
    """
    Perform fine-tuning for code stylometry classification using T5 encoder.
    """
    print(f"Processing file: {json_path}")

    # Load and filter data
    df = load_and_preprocess_data(json_path)
    if df.empty:
        print(f"Skipping file {json_path}: No valid classes with at least 2 samples.")
        return

    df = shuffle_dataframe(df)
    unique_authors = df["emailname"].unique()
    label_map = {author: idx for idx, author in enumerate(unique_authors)}

    # Ensure we have at least 2 classes
    if len(label_map) < 2:
        print(f"Skipping file {json_path}: Insufficient valid classes ({len(label_map)} < 2)")
        return

    # Train-test split
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["emailname"]
    )

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Encode data
    train_encodings, train_labels = preprocess_data(train_df, tokenizer, label_map)
    test_encodings, test_labels = preprocess_data(test_df, tokenizer, label_map)

    # Create datasets
    train_dataset = CodeStylometryDataset(train_encodings, train_labels)
    test_dataset = CodeStylometryDataset(test_encodings, test_labels)

    # Select and set GPU (or CPU fallback)
    gpu_id = select_gpu()
    if gpu_id > 0:
        torch.cuda.set_device(gpu_id)
        print(f"Using GPU: {gpu_id}")
    else:
        print("Using CPU.")

    # Build custom classification model
    model = T5ForCodeClassification(model_ckpt, num_labels=len(label_map)).to(
        "cuda" if gpu_id > 0 else "cpu"
    )

    # Create output directory per file
    model_output_dir = os.path.join(output_dir, os.path.basename(json_path).replace(".json", ""))

    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        weight_decay=0.01,
        save_strategy="no",
        logging_dir=f"{model_output_dir}/logs",
        logging_steps=10,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    early_stopping_callback = EarlyStoppingCallback(threshold_loss=0.03, patience=5)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
    )

    # Training & evaluation
    try:
        trainer.train()
        eval_results = trainer.evaluate()

        predictions = trainer.predict(test_dataset).predictions
        predicted_labels = np.argmax(predictions, axis=1)

        # Save or print evaluation results
        save_evaluation_results(
            eval_results,
            predicted_labels,
            test_labels,
            test_df,
            train_df,
            label_map,
            model_output_dir
        )

    except Exception as e:
        print(f"Critical error processing {json_path}: {str(e)}")
        raise

    print(f"Finished processing file: {json_path}")


##############################################################################
#                                   MAIN                                     #
##############################################################################
def main():
    """
    Main entry point for script execution.
    """

    json_dir = f"language/combined_{args.lang}"
    json_files = glob.glob(os.path.join(json_dir, "*.json"))

    if not json_files:
        print(f"No JSON files found in directory: {json_dir}")
        return

    for json_file in json_files:
        try:
            fine_tune(json_file, output_dir=f"./t5_{args.lang}")
        except Exception as e:
            print(f"Error processing file {json_file}: {str(e)}")


if __name__ == "__main__":
    main()
