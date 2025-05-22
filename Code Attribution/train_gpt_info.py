import json
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm
import pandas as pd
from openai import OpenAI

# Configuration
parser = argparse.ArgumentParser(description="OpenAI Authorship Attribution")
parser.add_argument("--lang", type=str, required=True, help="Programming language of the dataset")
parser.add_argument("--model", type=str, default="gpt-4", help="OpenAI model to use")
args = parser.parse_args()

# Initialize the OpenAI client with the new API format
client = OpenAI(api_key='')

# Load prompt template
with open('prompt_info.txt', 'r') as f:
    PROMPT_TEMPLATE = f.read()

def process_repository(repo_path):
    # Load and preprocess data from JSON
    with open(repo_path, 'r') as f:
        data = json.load(f)
    
    # Process records: include added_code, message, and filename if any exist
    records = []
    for item in data:
        # Ensure added_code is a string (join if it's a list)
        added_code = "\n".join(item.get("added_code", [])) if isinstance(item.get("added_code"), list) else item.get("added_code", "")
        message = item.get("message", "")
        filename = item.get("filename", "")
        emailname = item.get("emailname", "")
        
        # Only add the record if at least one of the fields is non-empty
        if added_code or message or filename:
            records.append({
                "added_code": added_code,
                "message": message,
                "filename": filename,
                "emailname": emailname
            })
    
    # Convert to DataFrame and filter for authors with at least 2 samples
    df = pd.DataFrame(records)
    author_counts = df['emailname'].value_counts()
    valid_authors = author_counts[author_counts >= 2].index.tolist()
    filtered_data = df[df['emailname'].isin(valid_authors)].to_dict('records')
    
    # Determine appropriate test_size ratio to ensure at least one sample per class
    total_samples = len(filtered_data)
    num_classes = len(valid_authors)
    default_test_ratio = 0.2
    # Calculate the number of test samples (round up)
    n_test_samples = int(np.ceil(total_samples * default_test_ratio))
    if n_test_samples < num_classes:
        test_ratio = num_classes / total_samples
    else:
        test_ratio = default_test_ratio

    # Split data into training and testing sets
    train_data, test_data = train_test_split(
        filtered_data, 
        test_size=test_ratio, 
        random_state=42,
        stratify=[item['emailname'] for item in filtered_data]
    )
    
    # Prepare few-shot examples that include filename, message, and code
    examples = "\n".join([
        f"Filename: {item['filename']}\nMessage: {item['message']}\nCode:\n{item['added_code']}\nAuthor: {item['emailname']}" 
        for item in train_data[:3]
    ])
    authors = list(valid_authors)

    results = {
        "predictions": [],
        "true_labels": [],
        "test_items": []
    }

    # Process each test sample
    for item in tqdm(test_data, desc="Processing test samples"):
        # Combine filename, message, and added_code into a single test input
        test_input = f"Filename: {item['filename']}\nMessage: {item['message']}\nCode:\n{item['added_code']}"
        prompt = PROMPT_TEMPLATE.format(
            language=args.lang.upper(),
            examples=examples,
            author_list=", ".join(authors),
            test_input=test_input
        )
        
        # Call the API using the new client format
        response = client.chat.completions.create(
            model=args.model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": "You are a code authorship analysis expert."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Parse response to extract the predicted author
        pred_author = response.choices[0].message.content.strip()
        true_author = item['emailname']
        
        # Store the results
        results['predictions'].append(pred_author)
        results['true_labels'].append(true_author)
        results['test_items'].append({
            "added_code": item['added_code'],
            "message": item['message'],
            "filename": item['filename'],
            "emailname": true_author
        })

    # Calculate overall accuracy
    results['accuracy'] = accuracy_score(results['true_labels'], results['predictions'])

    # Build output data with overall and per-author metrics
    output_data = {
        "accuracy": results['accuracy'],
        "predictions": results['predictions'],
        "true_labels": results['true_labels'],
        "detailed_metrics": {
            "per_author": {},
            "overall": {
                "f1": f1_score(results['true_labels'], results['predictions'], average='weighted'),
                "recall": recall_score(results['true_labels'], results['predictions'], average='weighted'),
                "precision": precision_score(results['true_labels'], results['predictions'], average='weighted')
            }
        }
    }

    # Compute per-author precision, recall, and f1 scores
    precision_vals, recall_vals, f1_vals, _ = precision_recall_fscore_support(
        results['true_labels'], 
        results['predictions'], 
        labels=authors
    )
    
    for idx, author in enumerate(authors):
        output_data['detailed_metrics']['per_author'][author] = {
            "precision": precision_vals[idx],
            "recall": recall_vals[idx],
            "f1": f1_vals[idx]
        }

    # Create output directory structure based on language and repository name
    base_name = os.path.splitext(os.path.basename(repo_path))[0]
    output_dir = os.path.join(f"gpt4_{args.lang}_info", base_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save the evaluation results as a JSON file
    with open(os.path.join(output_dir, "evaluation_results_info.json"), 'w') as f:
        json.dump(output_data, f, indent=2)

if __name__ == "__main__":
    json_dir = f"combined_{args.lang}"
    # Create the output directory if it doesn't exist
    os.makedirs(f"gpt4_{args.lang}_info", exist_ok=True)
    # List of already processed repositories (by filename without extension)
    exist_file = [f.split('.')[0] for f in os.listdir(f"gpt4_{args.lang}_info")] if os.path.exists(f"gpt4_{args.lang}_info") else []
    
    # Print the list of existing files for debugging
    if exist_file:
        print("Already processed repositories:", exist_file)
    
    for json_file in os.listdir(json_dir):
        if json_file.endswith('.json'):
            base_name = json_file.split('.')[0]
            if base_name not in exist_file:
                print(f"Processing: {json_file}")
                process_repository(os.path.join(json_dir, json_file))
            else:
                print(f'Already covered: {json_file}')