import csv
from datetime import datetime

def format_dates(row, date_columns):
    """Format date fields to 'YYYY-MM-DD'."""
    for col in date_columns:
        if row[col]:  # Check if the date field is not empty
            try:
                # Parse the date in 'DD.MM.YYYY' format and reformat to 'YYYY-MM-DD'
                row[col] = datetime.strptime(row[col], "%d.%m.%Y").strftime("%Y-%m-%d")
            except ValueError as e:
                print(f"Error parsing date in column '{col}': {row[col]} - {e}")
    return row

def process_csv(input_file, output_file):
    with open(input_file, mode='r', encoding='utf-8') as infile, open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        # Identify columns that contain "dat" in their names (case-insensitive)
        date_columns = [col for col in reader.fieldnames if "dat" in col.lower()]

        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            formatted_row = format_dates(row, date_columns)
            writer.writerow(formatted_row)

# Specify the input and output CSV file paths
input_csv = 'zamestnanci.csv'  # Your input CSV file
output_csv = 'output.csv'  # Output CSV file with formatted dates

process_csv(input_csv, output_csv)
