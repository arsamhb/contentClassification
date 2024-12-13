import pandas as pd
import re
import emoji

def remove_emojis(text):
    return emoji.replace_emoji(text, replace="")

def convert_persian_digits_to_english(text):
    persian_to_english_digits = str.maketrans("\u06F0\u06F1\u06F2\u06F3\u06F4\u06F5\u06F6\u06F7\u06F8\u06F9", "0123456789")
    arabic_to_english_digits = str.maketrans("\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669", "0123456789")
    text = text.translate(persian_to_english_digits)
    text = text.translate(arabic_to_english_digits)
    return text

def clean_caption(text):
    if not isinstance(text, str):
        return ""
    text = remove_emojis(text)
    text = text.lower()
    text = convert_persian_digits_to_english(text)

    def replace_special_chars(match):
        char = match.group()
        before = match.start() - 1  
        after = match.end()        
        if re.match(r"^@[\w\-]+$", text) or re.match(r"^www\.[\w\-]+\.[a-z]+$", text):
            return char
        if before >= 0 and after < len(text):
            if char in "_-" and text[before].strip() and text[after].strip():
                return char
        return " "

    # text = re.sub(r"[\n\t:\-_,/\\|?#؛▲…،!$%-^&ـ—•.*()\[\]{}<>~`+=;\'\"]", replace_special_chars, text)
    text = re.sub(r"[\n\t:\-_,/\\|?#!$%-]", replace_special_chars, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

file_name = "./../data/post_data/extracted_data_full.csv"
data = pd.read_csv(file_name)

data['caption'] = data['caption'].apply(clean_caption)

# Save the cleaned dataset
output_file_name = "./../data/post_data/extracted_data_full_cleaned.csv"
data[['caption']].to_csv(output_file_name, index=False)

print(f"Cleaned dataset saved as {output_file_name}")
