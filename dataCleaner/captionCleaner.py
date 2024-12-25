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

def normalize_text(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r'[ـ\-_]{2,}', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_farsi_repetitions(text):
    if not isinstance(text, str):
        return ""
    
    # Define a regex pattern for Farsi characters with repeated occurrences
    farsi_repetition_pattern = r'([\u0600-\u06FF])\1{2,}'
    
    # Replace repetitions of more than 2 characters with a single occurrence
    text = re.sub(farsi_repetition_pattern, r'\1', text)
    
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_caption(text):
    if not isinstance(text, str):
        print("asdljlashfilsehfusuilhf")
        return ""
    
    text = remove_emojis(text)
    text = text.lower()
    text = convert_persian_digits_to_english(text)
    
    def replace_special_chars(match):
        word = match.group()

        # Define patterns for IDs and websites
        id_pattern = r"@[\w\-.]+"
        website_pattern = r"\b((http://|https://|www\.)?[\w\-.]+\.[a-z]{2,}(/[\S]*)?)\b"

        extracted = []

        id_matches = re.findall(id_pattern, word)
        website_matches = re.findall(website_pattern, word)

        extracted.extend(id_matches)
        extracted.extend([m[0] for m in website_matches])

        for match in id_matches + [m[0] for m in website_matches]:
            word = word.replace(match, "")
        # ⚘ ᯐ ༄ ❀
        cleaned_word = re.sub(r"[:\-_/\\|?#!$%٫ঔ✞𓆪𓊆𓊇꧁⟢꧂𓆩☟♱☬𝜚𝜗◇\-&*()\[\]{˚⋆}<—•>ⓒ»«』✸『‐✯〗؛–☻°■“”─━〖✾؟~▲+=·.,^…،;ـ✰●★\'\"]", " ", word)
        cleaned_word = re.sub(r"\s+", " ", cleaned_word).strip()

        return " ".join(extracted + [cleaned_word]).strip()
    
    text = re.sub(r"[\S]+", replace_special_chars, text)

    text = normalize_text(text)

    text = normalize_farsi_repetitions(text)

    text = re.sub(r"\s+", " ", text).strip()
    return text

# file_name = "./../data/post_data/extracted_data_full.csv"
file_name = "./../data/test/test.csv"
data = pd.read_csv(file_name)

data['caption'] = data['caption'].apply(clean_caption)

# output_file_name = "./../data/post_data/extracted_data_full_cleaned.csv"
output_file_name = "./../data/test/test_cleaned.csv"
data[['caption']].to_csv(output_file_name, index=False)

print(f"Cleaned dataset saved as {output_file_name}")