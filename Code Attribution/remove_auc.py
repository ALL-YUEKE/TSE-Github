import os
import csv
import re

def process_result_files():
    """Process all CSV files in the result directory to remove AUC column"""
    result_dir = os.path.join(os.getcwd(), 'graph_result')
    
    # Check if result directory exists
    if not os.path.exists(result_dir):
        print(f"Result directory not found at {result_dir}")
        return
    
    # Process each CSV file in the result directory
    for filename in os.listdir(result_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(result_dir, filename)
            remove_auc_column(file_path)
            print(f"Processed: {filename}")

def remove_auc_column(file_path):
    """Remove the AUC column from a CSV file"""
    # Read the CSV file
    rows = []
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Remove the last column (AUC) if it exists
            if row and len(row) > 3:  # Ensure there are enough columns
                rows.append(row[:-1])
            else:
                rows.append(row)
    
    # Write the modified data back to the file
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)

def process_final_csv():
    """Remove the average_auc column from final.csv"""
    final_csv_path = os.path.join(os.getcwd(), 'final.csv')
    
    # Check if final.csv exists
    if not os.path.exists(final_csv_path):
        print(f"Final CSV not found at {final_csv_path}")
        return
    
    # Read the CSV file
    rows = []
    with open(final_csv_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Remove the average_auc column (4th column) if it exists
            if row and len(row) > 3:  # Ensure there are enough columns
                rows.append(row[:-1])
            else:
                rows.append(row)
    
    # Write the modified data back to the file
    with open(final_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)
    
    print("Processed: final.csv")

if __name__ == "__main__":
    print("Starting AUC column removal process...")
    process_result_files()
    print("AUC column removal completed!")