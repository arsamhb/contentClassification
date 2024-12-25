import re


# Function to convert English digits to Persian digits
def convert_to_persian_digits(text):
    english_to_persian = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
    return text.translate(english_to_persian)


# Function to process the input file
def convert_file_digits(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    # Convert English digits to Persian digits line by line
    converted_lines = [convert_to_persian_digits(line) for line in lines]

    # Write the converted lines to the output file
    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.writelines(converted_lines)


# File paths
input_file = "./../data/test/test_cleaned_text.txt"
output_file = "./../data/test/test_captions_persian_digits.txt"

# Process the file
convert_file_digits(input_file, output_file)

print(f"File successfully converted and saved to {output_file}")
