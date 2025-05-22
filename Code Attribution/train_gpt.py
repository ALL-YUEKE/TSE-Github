import openai
import json
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm
import pandas as pd
# Configuration
parser = argparse.ArgumentParser(description="OpenAI Authorship Attribution")
parser.add_argument("--lang", type=str, required=True, help="Programming language of the dataset")
parser.add_argument("--model", type=str, default="gpt-4", help="OpenAI model to use")
args = parser.parse_args()

# Initialize OpenAI client
client = openai.OpenAI(api_key='')

# Load prompt template
with open('attribution_prompt.txt', 'r') as f:
    PROMPT_TEMPLATE = f.read()

def process_repository(repo_path):
    # Load and preprocess data
    with open(repo_path, 'r') as f:
        data = json.load(f)
    
    # Filter authors with minimum 2 samples
    df = pd.DataFrame(data)
    author_counts = df['emailname'].value_counts()
    valid_authors = author_counts[author_counts >= 2].index.tolist()
    filtered_data = df[df['emailname'].isin(valid_authors)].to_dict('records')

    # Split data
    train_data, test_data = train_test_split(
        filtered_data, 
        test_size=0.2, 
        random_state=42,
        stratify=[item['emailname'] for item in filtered_data]
    )
    
    # Prepare few-shot examples
    examples = "\n".join([f"Code:\n{item['added_code']}\nAuthor: {item['emailname']}" 
                    for item in train_data[:3]])
    authors = list(valid_authors)

    results = {
        "predictions": [],
        "true_labels": [],
        "test_items": []
    }

    # Process test samples
    for item in tqdm(test_data, desc="Processing test samples"):
        code = "\n".join(item['added_code'])
        prompt = PROMPT_TEMPLATE.format(
            language=args.lang.upper(),
            examples=examples,
            author_list=", ".join(authors),
            test_code=code
        )
        
        # Get OpenAI prediction
        response = client.chat.completions.create(
            model=args.model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": "You are a code authorship analysis expert."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Parse response
        pred_author = response.choices[0].message.content.strip()
        true_author = item['emailname']
        
        # Store results
        results['predictions'].append(pred_author)
        results['true_labels'].append(true_author)
        results['test_items'].append({
            "added_code": item['added_code'],
            "emailname": true_author
        })

    # Calculate accuracy
    results['accuracy'] = accuracy_score(results['true_labels'], results['predictions'])

    # Create output structure
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

    # Calculate per-author metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        results['true_labels'], 
        results['predictions'], 
        labels=authors
    )
    
    for idx, author in enumerate(authors):
        output_data['detailed_metrics']['per_author'][author] = {
            "precision": precision[idx],
            "recall": recall[idx],
            "f1": f1[idx]
        }

    # Create output directory structure
    base_name = os.path.splitext(os.path.basename(repo_path))[0]
    output_dir = os.path.join(f"gpt4_{args.lang}", base_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save results
    with open(os.path.join(output_dir, "evaluation_results.json"), 'w') as f:
        json.dump(output_data, f, indent=2)

if __name__ == "__main__":
    json_dir = f"combined_{args.lang}"
    exist_file=[f.split('.')[0] for f in os.listdir(f"gpt4_{args.lang}")]  
    for json_file in os.listdir(json_dir):
        if json_file.endswith('.json'):
            if json_file.split('.')[0] not in exist_file:
                process_repository(os.path.join(json_dir, json_file))
