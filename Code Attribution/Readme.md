# Code Authorship Attribution Project
This repository contains the code and datasets for our research on code authorship attribution using deep learning models. Our work focuses on identifying the true authors of code commits across five major programming languages.

## Project Overview
Our research makes several key contributions:

1. Large-Scale Dataset Construction : We started with 12.5 million commits and, through rigorous filtering, curated a specialized dataset of 583,194 verified commits from 370 repositories across five major programming languages (Go, Java, JavaScript, PHP, and Python).
2. Extensive Model Evaluation : We assessed multiple deep learning models including CodeBERT, GraphCodeBERT, CodeT5+, and GPT-4o-mini to determine their effectiveness in identifying code authors.
3. Novel Input Representation Analysis : We compared a baseline "Code-Only" input with a "Comprehensive Information" (Com-Info) approach that incorporates commit metadata, demonstrating significant accuracy improvements.
4. In-depth Analysis of Impersonation Detection : We explored the challenges in deterministic impersonation detection due to Git mechanisms and pull request workflows.
   


## Installation
# Clone the repository
git clone https://github.com/yourusername/code-authorship-attribution.git
cd code-authorship-attribution

# Install dependencies
pip install torch transformers pandas numpy scikit-learn tqdm openai

## Usage
### Training Models Basic Models (Code-Only)
# Train CodeBERT model
python train_codebert.py --lang python

# Train GraphCodeBERT model
python train_graphcodebert.py --lang java

# Train CodeT5+ model
python train_t5.py --lang javascript

# Train with GPT-4
python train_gpt.py --lang php --model gpt-4

### Comprehensive Information Models
# Train CodeBERT with comprehensive information
python train_codebert_info.py --lang python

# Train GraphCodeBERT with comprehensive information
python train_graphcodebert_info.py --lang go

# Train CodeT5+ with comprehensive information
python train_t5_info.py --lang javascript
