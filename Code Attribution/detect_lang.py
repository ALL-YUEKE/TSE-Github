import os
import json
import argparse
from tqdm import tqdm
from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound
import pandas as pd

def detect_language(code_snippet):
    extensions_to_languages = {
        # General Programming Languages
        '.py': 'Python',
        '.js': 'JavaScript',
        '.ts': 'TypeScript',
        '.java': 'Java',
        '.c': 'C',
        '.cpp': 'C++',
        '.cs': 'C#',
        '.rb': 'Ruby',
        '.php': 'PHP',
        '.go': 'Go',
        '.swift': 'Swift',
        '.kt': 'Kotlin',
        '.rs': 'Rust',
        '.dart': 'Dart',
        '.lua': 'Lua',
        '.pl': 'Perl',
        '.r': 'R',
        '.sh': 'Shell Script',
        '.m': 'MATLAB/Objective-C',
        '.f90': 'Fortran',
        '.jl': 'Julia',
        '.hs': 'Haskell',
        '.erl': 'Erlang',
        '.ex': 'Elixir',
        '.lisp': 'Common Lisp',
        '.clj': 'Clojure',
        '.coffee': 'CoffeeScript',
        '.scala': 'Scala',
        '.groovy': 'Groovy',
        '.v': 'Verilog',
        '.sv': 'SystemVerilog',
        '.ml': 'OCaml',
        '.vbs': 'VBScript',
        '.adb': 'Ada',
        '.awk': 'AWK',
        '.nim': 'Nim',
        '.cr': 'Crystal',
        '.pas': 'Pascal',
        '.d': 'D',
        
        # Scripting and Configurations
        '.bat': 'Batch File',
        '.ps1': 'PowerShell',
        '.yaml': 'YAML',
        '.yml': 'YAML',
        '.toml': 'TOML',
        '.ini': 'INI',
        '.json': 'JSON',
        '.xml': 'XML',
        '.csv': 'Comma-Separated Values',
        '.env': 'Environment Configuration',

        # Markup and Web
        '.html': 'HTML',
        '.htm': 'HTML',
        '.xhtml': 'XHTML',
        '.md': 'Markdown',
        '.css': 'CSS',
        '.scss': 'Sass',
        '.less': 'LESS',

        # Database and Queries
        '.sql': 'SQL',
        '.psql': 'PostgreSQL',
        '.sqlite': 'SQLite',
        '.db': 'Database',

        # Data Science and Notebooks
        '.ipynb': 'Jupyter Notebook',
        '.rmd': 'R Markdown',
        '.sas': 'SAS',
        '.sav': 'SPSS',
        '.dta': 'Stata',
        
        # Embedded Systems and Low-Level
        '.asm': 'Assembly',
        '.s': 'Assembly',
        '.hex': 'Intel HEX',
        '.ino': 'Arduino',

        # Functional Programming
        '.fs': 'F#',
        '.fsx': 'F# Script',
        '.elm': 'Elm',
        
        # Miscellaneous
        '.apk': 'Android Package',
        '.exe': 'Windows Executable',
        '.bin': 'Binary File',
        '.so': 'Shared Object',
        '.dll': 'Dynamic-Link Library',

        # Testing and Build Tools
        '.test': 'Test File',
        '.spec': 'Test Specification',
        '.travis.yml': 'Travis CI Configuration',
        '.makefile': 'Makefile',
        '.cmake': 'CMake',

        # Logs and Documentation
        '.log': 'Log File',
        '.txt': 'Plain Text',
        '.rst': 'reStructuredText',
        
        # Graphics and Design
        '.svg': 'Scalable Vector Graphics',
        '.psd': 'Photoshop Document',
        '.ai': 'Adobe Illustrator File',
    }

    # Extract the file extension
    _, ext = os.path.splitext(code_snippet)

    # Return the corresponding language or a default message
    return extensions_to_languages.get(ext.lower(), "Unknown Language")

def process_files(csv_file, directory, start, end):
    """
    Processes JSON files listed in a CSV file, detects programming language
    in `added_code`, and adds the detected language to the JSON data.
    
    Args:
        csv_file (str): Path to the CSV file.
        directory (str): Directory containing JSON files.
        start (int): Start index for file processing.
        end (int): End index for file processing.
    """
    # Read the CSV file and extract filenames
    df = pd.read_csv(csv_file)
    filenames = df['filename'][start:end].str.replace('.csv', '', regex=False)

    for filename in tqdm(filenames, desc="Processing files"):
        json_path = os.path.join(directory, f"{filename}.json")
        if not os.path.exists(json_path):
            continue  # Skip if the JSON file is missing

        # Read the JSON file
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)

        # Process each item in the JSON file
        for item in data:
            added_code = item.get("filename")
            if added_code:
                # Combine all added_code snippets to guess the language
                language = detect_language(added_code)
                print(language)
            else:
                language = "Unknown"
            # Add the detected language to the item
            item["language"] = language

        # Save the updated JSON back to the file
        with open(json_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process JSON files listed in a CSV file to detect programming languages in added_code.")
    parser.add_argument("--start", type=int, default=0, help="Start index (default: 0)")
    parser.add_argument("--end", type=int, default=None, help="End index (default: None)")
    parser.add_argument("--csv_file", type=str, default="../all_author_commits3.csv", help="Path to the CSV file (default: all_author_commits3.csv)")
    parser.add_argument("--directory", type=str, default="commit_content", help="Directory containing JSON files (default: commit_content)")
    
    args = parser.parse_args()

    # Process the files
    process_files(args.csv_file, args.directory, args.start, args.end)

