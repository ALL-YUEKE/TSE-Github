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
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModel,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed,
    T5ForConditionalGeneration
)
from transformers.modeling_outputs import SequenceClassifierOutput
parser = argparse.ArgumentParser(description="Multi-Modal Fine-tuning with CodeT5 + DistilBERT")
parser.add_argument("--lang", type=str, required=True, help="Language code to process.")
args = parser.parse_args()

warnings.filterwarnings("ignore")


##############################################################################
#                              DATASET CLASS                                 #
##############################################################################
class MultiModalDataset(torch.utils.data.Dataset):
    """
    Multi-modal dataset that includes:
      1) code (tokenized by CodeT5),
      2) commit message (tokenized by DistilBERT),
      3) filename (also tokenized by DistilBERT),
      4) labels (encoded via LabelEncoder).
    """

    def __init__(self, df, code_tokenizer, text_tokenizer, label_encoder):
        """
        Args:
            df (pd.DataFrame): DataFrame with columns:
                - added_code
                - message
                - filename
                - emailname
            code_tokenizer (AutoTokenizer): Tokenizer for code (CodeT5).
            text_tokenizer (AutoTokenizer): Tokenizer for DistilBERT (used for message & filename).
            label_encoder (LabelEncoder): Encodes author labels into integer IDs.
        """

        # --- 1) Code tokenization ---
        self.code_encodings = code_tokenizer(
            df["added_code"].tolist(),
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )

        # --- 2) Commit message tokenization ---
        self.message_encodings = text_tokenizer(
            df["message"].tolist(),
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )

        # --- 3) Filename tokenization ---
        self.filename_encodings = text_tokenizer(
            df["filename"].tolist(),
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )

        # --- 4) Label encoding ---
        self.labels = torch.tensor(
            label_encoder.transform(df["emailname"]),
            dtype=torch.long
        )

    def __getitem__(self, idx):
        """
        Return a single example as a dictionary that includes tokenized code,
        message, filename, and the integer label.
        """
        return {
            # Code
            "code_input_ids": self.code_encodings["input_ids"][idx],
            "code_attention_mask": self.code_encodings["attention_mask"][idx],
            # Message
            "message_input_ids": self.message_encodings["input_ids"][idx],
            "message_attention_mask": self.message_encodings["attention_mask"][idx],
            # Filename
            "filename_input_ids": self.filename_encodings["input_ids"][idx],
            "filename_attention_mask": self.filename_encodings["attention_mask"][idx],
            # Label
            "labels": self.labels[idx]
        }

    def __len__(self):
        """
        Number of examples in the dataset.
        """
        return len(self.labels)


##############################################################################
#                           METRICS & CALLBACKS                               #
##############################################################################
def compute_metrics(eval_pred):
    """
    Compute the accuracy metric for evaluation.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}


class EarlyStoppingCallback(TrainerCallback):
    """
    Simple early stopping callback based on eval_loss.
    """
    def __init__(self, threshold_loss=0.1, patience=3):
        """
        Args:
            threshold_loss (float): If eval_loss goes below this, stop training.
            patience (int): If eval_loss does not improve for 'patience' epochs, stop training.
        """
        self.threshold_loss = threshold_loss
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def on_evaluate(self, args, state, control, **kwargs):
        eval_loss = kwargs["metrics"].get("eval_loss")
        if eval_loss is not None:
            # 1) Check absolute threshold
            if eval_loss < self.threshold_loss:
                print(f"Stopping early: eval_loss={eval_loss:.4f} < threshold {self.threshold_loss}")
                control.should_training_stop = True
            # 2) Check improvement
            elif eval_loss < self.best_loss:
                self.best_loss = eval_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"Stopping early: eval_loss did not improve for {self.patience} evaluations.")
                    control.should_training_stop = True


##############################################################################
#                           MULTI-MODAL MODEL                                #
##############################################################################
class MultiModalModel(nn.Module):
    """
    A multi-modal model that uses:
      - CodeT5 for code
      - DistilBERT for commit message
      - DistilBERT for filename
    Then concatenates all pooled outputs for classification.
    """

    def __init__(
        self,
        code_model_ckpt,
        text_model_ckpt,
        num_labels
    ):
        """
        Args:
            code_model_ckpt (str): HF model name/path for CodeT5 (e.g., "Salesforce/codet5p-220m").
            text_model_ckpt (str): HF model name/path for DistilBERT (used for message & filename).
            num_labels (int): Number of distinct author classes.
        """
        super().__init__()

        self.num_labels = num_labels

        # 1) Code encoder (CodeT5).
        #    We'll use T5ForConditionalGeneration but only call the encoder.
        self.code_encoder_model = T5ForConditionalGeneration.from_pretrained(code_model_ckpt)

        # 2) DistilBERT for commit message
        self.message_encoder = AutoModel.from_pretrained(text_model_ckpt)

        # 3) DistilBERT for filename
        self.filename_encoder = AutoModel.from_pretrained(text_model_ckpt)

        # Hidden sizes
        # T5: we have to look at self.code_encoder_model.config.d_model
        code_hidden_size = self.code_encoder_model.config.d_model
        msg_hidden_size = self.message_encoder.config.hidden_size
        file_hidden_size = self.filename_encoder.config.hidden_size

        # Final classification head
        self.fc = nn.Sequential(
            nn.Linear(code_hidden_size + msg_hidden_size + file_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_labels)
        )

    def forward(
        self,
        code_input_ids=None,
        code_attention_mask=None,
        message_input_ids=None,
        message_attention_mask=None,
        filename_input_ids=None,
        filename_attention_mask=None,
        labels=None
    ):
        # --- 1) T5 Encoder output (CodeT5) ---
        encoder_outputs = self.code_encoder_model.encoder(
            input_ids=code_input_ids,
            attention_mask=code_attention_mask,
            return_dict=True
        )
        # shape: (batch_size, seq_len, d_model)
        code_hidden_states = encoder_outputs.last_hidden_state
        # Mean-pooling to get a single vector
        code_pooled = code_hidden_states.mean(dim=1)

        # --- 2) DistilBERT for commit message ---
        msg_outputs = self.message_encoder(
            input_ids=message_input_ids,
            attention_mask=message_attention_mask
        )
        msg_pooled = msg_outputs.last_hidden_state[:, 0, :]

        # --- 3) DistilBERT for filename ---
        file_outputs = self.filename_encoder(
            input_ids=filename_input_ids,
            attention_mask=filename_attention_mask
        )
        file_pooled = file_outputs.last_hidden_state[:, 0, :]

        # --- Concatenate all three pooled vectors ---
        combined = torch.cat((code_pooled, msg_pooled, file_pooled), dim=1)

        # --- Classification Head ---
        logits = self.fc(combined)

        # --- Compute Loss (if labels are provided) ---
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)


##############################################################################
#                      COLLATE FUNCTION FOR DATALOADER                       #
##############################################################################
def collate_fn(batch):
    """
    Custom collate function to batch data from MultiModalDataset.
    """
    if not batch:
        raise ValueError("Received an empty batch.")

    return {
        "code_input_ids": torch.stack([item["code_input_ids"] for item in batch]),
        "code_attention_mask": torch.stack([item["code_attention_mask"] for item in batch]),
        "message_input_ids": torch.stack([item["message_input_ids"] for item in batch]),
        "message_attention_mask": torch.stack([item["message_attention_mask"] for item in batch]),
        "filename_input_ids": torch.stack([item["filename_input_ids"] for item in batch]),
        "filename_attention_mask": torch.stack([item["filename_attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }


##############################################################################
#                   EVALUATION RESULTS SAVING (CUSTOM)                       #
##############################################################################
def save_evaluation_results(eval_results, predictions, labels, test_df, output_dir, label_encoder):
    """
    Saves detailed evaluation results (accuracy, predictions, true labels, etc.)
    to a JSON file "evaluation_results_info.json" in the output directory.
    """
    from sklearn.metrics import classification_report

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
            "test_items": test_df.to_dict(orient="records"),
        }

        output_path = os.path.join(output_dir, "evaluation_results_info.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        print(f"Evaluation results saved to {output_path}")

    except Exception as e:
        print(f"Error saving evaluation results: {e}")


##############################################################################
#                             GPU SELECTION HELPER                            #
##############################################################################
def select_gpu(min_free_mem_mb=10000, max_free_mem_mb=80000):
    """
    Utility to select a GPU device within certain free-memory constraints.
    If none found, default to GPU 0 or raise an error.
    """
    device_count = torch.cuda.device_count()
    if device_count < 1:
        raise RuntimeError("No GPU found on this machine.")

    for gpu_id in range(device_count):
        props = torch.cuda.get_device_properties(gpu_id)
        used_mem = torch.cuda.memory_allocated(gpu_id)
        free_mem = (props.total_memory - used_mem) // (1024 * 1024)
        if min_free_mem_mb < free_mem < max_free_mem_mb:
            return gpu_id
    return 0  # Fallback if none match criteria


##############################################################################
#                             MAIN TRAINING LOOP                              #
##############################################################################
def fine_tune(
    json_path,
    code_model_ckpt="Salesforce/codet5p-220m",
    text_model_ckpt="distilbert-base-uncased",
    output_dir=f"./codet5_{args.lang}_info",
    seed=42
):
    """
    Fine-tune a multi-modal model that uses:
      - CodeT5 for code embeddings
      - DistilBERT for commit message
      - DistilBERT for filename.
    """
    print(f"Processing file: {json_path}")

    # Set seeds for reproducibility
    set_seed(seed)

    # Load data
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file {json_path}: {e}")
        return

    # Build records list
    records = []
    for item in data:
        added_code = "\n".join(item.get("added_code", [])) if item.get("added_code") else ""
        message = item.get("message", "")
        filename = item.get("filename", "")
        emailname = item.get("emailname", "")

        # Only add rows if there's at least some code, message, or filename
        if added_code or message or filename:
            records.append({
                "added_code": added_code,
                "message": message,
                "filename": filename,
                "emailname": emailname
            })

    df = pd.DataFrame(records)
    if df.empty:
        print(f"No valid samples in {json_path}. Skipping.")
        return

    # Filter out authors with fewer than 2 samples, for example
    class_counts = df["emailname"].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    df = df[df["emailname"].isin(valid_classes)]
    if df.empty:
        print(f"No valid classes with >=2 samples in {json_path}. Skipping.")
        return

    # Shuffle data
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Label encoding
    label_encoder = LabelEncoder()
    label_encoder.fit(df["emailname"])

    # Train/test split
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=seed,
            stratify=df["emailname"]
        )
    except ValueError as e:
        print(f"Stratified split error for {json_path}: {e}")
        return

    # Initialize tokenizers
    code_tokenizer = AutoTokenizer.from_pretrained(code_model_ckpt)
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_ckpt)

    # Create datasets
    train_dataset = MultiModalDataset(train_df, code_tokenizer, text_tokenizer, label_encoder)
    test_dataset = MultiModalDataset(test_df, code_tokenizer, text_tokenizer, label_encoder)

    # Select GPU
    try:
        gpu_id = select_gpu()
        torch.cuda.set_device(gpu_id)
        print(f"Using GPU: {gpu_id}")
    except RuntimeError as e:
        print(e)
        return

    # Prepare output directory
    model_output_dir = os.path.join(output_dir, os.path.basename(json_path).replace(".json", ""))
    os.makedirs(model_output_dir, exist_ok=True)

    # Model initialization
    num_labels = len(label_encoder.classes_)
    model = MultiModalModel(
        code_model_ckpt=code_model_ckpt,
        text_model_ckpt=text_model_ckpt,
        num_labels=num_labels
    ).cuda()

    # Define training arguments
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
        seed=seed
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        tokenizer=code_tokenizer,  # for special tokens or logging
        callbacks=[EarlyStoppingCallback(threshold_loss=0.1, patience=3)]
    )

    # Training and evaluation
    try:
        trainer.train()
        eval_results = trainer.evaluate()
        predictions = trainer.predict(test_dataset).predictions
        predicted_labels = np.argmax(predictions, axis=1)

        # Save or print detailed evaluation results
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


##############################################################################
#                                MAIN ENTRY                                   #
##############################################################################
def main():

    json_dir = f"language/combined_{args.lang}"
    json_files = glob.glob(os.path.join(json_dir, "*.json"))

    if not json_files:
        print(f"No JSON files found in directory: {json_dir}")
        return

    for json_file in json_files:
        fine_tune(
            json_file,
            code_model_ckpt="Salesforce/codet5p-220m",
            text_model_ckpt="distilbert-base-uncased",
            output_dir=f"./codet5_{args.lang}_info",
            seed=42
        )


if __name__ == "__main__":
    main()
