# PRICE
price_prefix_terms = [
    "قی",
    "قي",
    "قیمت",
    "هزینه",
    "هزينه",
    "بها",
    "في",
    "ق",
    "قیم",
    "قيم",
    "قی مت",
    "قي مت",
    "بهاي",
    "تک",
    "قیمتش",
    "قيمتش",
    "عمده",
    "مبلغ",
    "فقط",
    "قيمت",
]
price_appendix_terms = [
    "ت",
    "تومان",
    "تومن",
    "ریال",
    "t",
    "تو",
    "هزار",
    "دلار",
]

import re
import pandas as pd
import unicodedata

# IN THIS CASE WE CAN EXTEND THE PREV NEIBOUR SIZE AND THEN CONSIDER USING SOME METRICS LIKE METRI CANTI GRAMI BETWEEN THE
# PREVIOUSE IDENTIFIER AND THE DIGIT VALUE ITSELF
# قیمت O
# بااحترام O
# متری O
# 980 O

# LETS MAKE A LIST FROM EXCEPTIONS AND PUT تایی IN IT SO WE SAY IF THE APPENDIX AFTER THE DIGIT WAS IT DO NOT LABEL IT AS
# PRICE AND JUST PASS IT AS O
# عدد O
# درجین O
# 45 B-PRICE
# تومان O
# قیمت O
# جین O
# 12 B-PRICE
# تایی O
# 540 B-PRICE
# تومان O
# تک O

# WE CAN LOOK FOR SEQUENCES THAT 2 300 000 OR 2300 THE COMMON POINT IN THEM IS THAT THEY HAVE MORE AT LEAST 3 DIGITS
# THE OTHER COMMON POINT IS THAT THEY ARE A SEQUENCE OF ONLY-DIGIT-TOKENS OR A SINGLE ONLY-DIGIT-TOKEN WITH AT LEAST 3 DIGITS 
# ۲.۷۰۰

# THIS IS ANOTHER SCENARIO IF WE SEE A ONLY-DIGIT-TOEKN RIGHT BEFORE A TOKEN WITH PRICE TERM APPENDIX WE TAG IT AS B-PRICE 
# کرک O
# 798 O
# 000t B-PRICE
# free O
# size O
# ta O
# I-PRICE IS ONLY ALLOWED AFTER A B-PRICE
# 44 I-PRICE
# قد O
# 67 O

# HERE WE CAN MAKE EXCEPTION PREFIX TERMS LIST AND PUT ارسال IN IT AND A FULL-DIGIT-TOKEN MUST NOT BECOME AFTER OR NEIBOURING
# CLOSE TO THEM
# هزینه O
# ارسال O
# 45 B-PRICE
# ت O
# ارسال O

# WHEN WE FIND A B-PRICE WE GO FORWARD TO MARK ALL THE I-PRICES AND THE ITERATOR MUST GO ON
# AFTER THE LAST TOKEN WE CHECKED
#  O
# قیمت O
# 5 B-PRICE
# 640 B-PRICE
# تومن O
# قیمت O
# با O
# تخفیف O
# ویژه O
# 4 B-PRICE
# 640 B-PRICE
# تومن O
# سفارش O
# سریع

#  WE CAN IGNORE THIS FOR NOW BUT WHEN WE HAD A EXTRACTED OTHER FEATURES VALUES LIKE THIS CAN BE LABELED AS PRICE
# اقتصادی O
# باکیفیت208تومن B-PRICE
# شرتک O
# 89 I-PRICE
# نیمتنه O
# 119 I-PRICE
# نیمتنه O
# فری

# WE NEED TO ADD با - احترام TO THE PRICE PREFIXES LIST
# وارداتی O
# با O
# احترام680 O
# سایزبندی O
# مناسب O

# we need to take the unit of the price rial toman dollar as the PRICE-I SO I GUESS WE NEED A PRICE UNIT LIST TOO

def clean_token(token):
    token = unicodedata.normalize("NFKC", token)    
    token = token.replace("\u200c", "")
    token = ''.join(c for c in token if unicodedata.category(c) != 'Mn')
    
    return token


def is_price_token(token, prefix_terms, appendix_terms):
    match = re.match(r"^(\D+)?(\d+)(\D+)?$", token)
    if match:
        before = match.group(1) or ""
        after = match.group(3) or ""
        non_digit_parts = (before.strip(), after.strip())
        
        if non_digit_parts[0] in prefix_terms or non_digit_parts[1] in appendix_terms:
            return True
    
    prefix_regex = rf"^({'|'.join(prefix_terms)})\d+$"
    appendix_regex = rf"^\d+({'|'.join(appendix_terms)})$"
    combined_regex = rf"^({'|'.join(prefix_terms)})\d+({'|'.join(appendix_terms)})$"
    
    if (
        re.match(prefix_regex, token)
        or re.match(appendix_regex, token)
        or re.match(combined_regex, token)
    ):
        return True
    
    return False

def tokenize_and_label(text):
    tokens = re.findall(r"\S+|\n", text)
    labels = ["O"] * len(tokens)

    tokens = [clean_token(t) for t in tokens]

    for i, token in enumerate(tokens):
        if not re.search(r"\d", token):
            continue

        direct_match = is_price_token(token, price_prefix_terms, price_appendix_terms)

        if direct_match:
            labels[i] = "B-PRICE"
            j = i + 1
            if j > len(tokens):
                continue
            while j < len(tokens):
                if re.match(r"^\d+$", tokens[j]):
                    labels[j] = "I-PRICE"
                else:
                    match = re.match(r"(\d+)(\D+)", tokens[j])
                    if match:
                        non_digits = match.group(2).strip()

                        if non_digits in price_appendix_terms:
                            labels[j] = "I-PRICE"
                j += 1
            continue

        nearby_previous_context = tokens[max(0, i - 2) : i]
        nearby_next_context = tokens[i + 1 : min(len(tokens), i + 3)]

        has_prefix = any(term in nearby_previous_context for term in price_prefix_terms)
        has_appendix = any(term in nearby_next_context for term in price_appendix_terms)

        if has_prefix or has_appendix:
            mixed_match = False
            match = re.match(r"^(\D+)?(\d+)(\D+)?$", token)
            if match:
                before = (match.group(1) or "").strip()
                after = (match.group(3) or "").strip()

                if (
                    before == ""
                    or before in price_prefix_terms
                ) and (
                    after == ""
                    or after in price_appendix_terms
                ):
                    mixed_match = True
            else:
                if re.match(r"^\d+$", token):
                    mixed_match = True

            if mixed_match:
                labels[i] = "B-PRICE"
                j = i + 1
                while j < len(tokens) and re.match(r"^\d+$", tokens[j]):
                    labels[j] = "I-PRICE"
                    j += 1
            else:
                pass

    return list(zip(tokens, labels))


def process_dataset(file_path, labeled_output_path, original_output_path):
    data = pd.read_csv(file_path, header=None, names=["text"])

    random_rows = data.sample(n=10)

    random_rows.to_csv(
        original_output_path, index=False, header=False, encoding="utf-8"
    )

    labeled_data = []
    for _, row in random_rows.iterrows():
        text = row["text"]
        tokenized_and_labeled = tokenize_and_label(text)
        labeled_data.extend(
            tokenized_and_labeled + [("", "")]
        )

    with open(labeled_output_path, "w", encoding="utf-8") as f:
        for token, label in labeled_data:
            if token:
                f.write(f"{token} {label}\n")
            else:
                f.write("\n")


input_file = "./../data/post_data/extracted_data_full_cleaned.csv"
output_file = "./../data/post_data/labeled_captions_test.txt"
original_output_file = "./../data/post_data/original_captions_test.txt"

process_dataset(input_file, output_file, original_output_file)
