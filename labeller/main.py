import re
import pandas as pd
import unicodedata

# PRICE
price_prefix_terms = [
    "قی",
    "قي",
    "قیمت",
    "ارزش",
    "هزینه",
    "هزينه",
    "بها",
    "قمت",
    "في",
    "ق",
    "قیم",
    "قیممت",
    "قيم",
    "قی مت",
    "price",
    "قي مت",
    "مت",
    "بهاي",
    "قیمتش",
    "قيمتش",
    "عمده",
    "مبلغ",
    "فقط",
    "قيمت",
    "احترام",
    "ت",
    "عدد",
    "بهای",
    "فقطط",
    "فروش",
]
price_appendix_terms = [
    "ت",
    "تومان",
    "تومن",
    "ریال",
    "t",
    "تو",
    "هزار",
    "$",
    "دلار",
    "توومان",
    "میلیون",
    "€",
    "¥",
]
delivery_terms = [
    "ارسال",
    "پست",
]
price_units_list = [
    "ت",
    "تومان",
    "تومن",
    "ریال",
    "t",
    "تو",
    "هزار",
    "دلار",
    "$",
    "میلیون",
    "توومان",
    "€",
]


def create_regex_pattern(special_char_list):
    return f"[{''.join(map(re.escape, special_char_list))}]"


def clean_token(token):
    token = unicodedata.normalize("NFKC", token)
    token = token.replace("\u200c", "")
    token = "".join(c for c in token if unicodedata.category(c) != "Mn")

    return token


def does_it_contain_price_prefix(token, prefix_terms):
    prefix_pattern = "|".join(map(re.escape, prefix_terms))
    prefix_regex = rf"({prefix_pattern})\d+"

    if re.search(prefix_regex, token):
        return True
    return False


def does_it_contain_price_appendix(token, appendix_terms):
    appendix_pattern = "|".join(map(re.escape, appendix_terms))
    appendix_regex = rf"\d+({appendix_pattern})"

    if re.search(appendix_regex, token):
        return True
    return False


def is_pure_digits(token: str) -> bool:
    return token.isdigit()


def get_neibouring(tokens, index, neibouring_size):
    before = tokens[max(0, index - neibouring_size) : index]
    after = tokens[index + 1 : min(len(tokens), index + 1 + neibouring_size - 1)]
    return before, after


def is_it_phone_number(token):
    phone_number_starter = ["021", "+98", "091", "093", "090", "099"]
    return any(token.startswith(starter) for starter in phone_number_starter)


def is_it_irrelevant_digit(prev_token, next_token=None):
    irrelevant_terms_list_prefix = [
        "سایز",
        "ارسال",
        "تیپاکس",
        "روزه",
        "پیشتاز",
        "پست",
        "شهرستان",
        "کشور",
        "سراسر",
        "کد",
        "ابعاد",
        "کمر",
        "سایزبندی",
        "قد",
        "فاق",
        "وزن",
        "حضوری",
        "شومیز",
        "رویه",
        "آستین",
        "بلندی",
        "عرض",
        "پهنا",
        "رنگ",
    ]
    irrelevant_terms_list_appendix = [
        "سایز",
        "روزه",
        "کد",
        "سانت",
        "سانتی",
        "پهنا",
        "سانتیمتر",
        "سانتی‌متر",
        "cm",
        "شومیز",
        "رویه",
        "آستین",
        "بلندی",
        "عرض",
        "gr",
        "گرم",
        "کامنت",
        "رنگ",
    ]
    return (
        prev_token in irrelevant_terms_list_prefix
        or next_token in irrelevant_terms_list_appendix
    )


def does_token_contain_irrelevant_term(token):
    irrelevant_terms = [
        "ارسال",
        "تیپاکس",
        "روزه",
        "پست",
        "کشور",
        "سراسر",
        "کد",
        "ابعاد",
        "کمر",
        "سایزبندی",
        "قد",
        "فاق",
        "وزن",
        "حضوری",
        "شومیز",
        "رویه",
        "آستین",
        "بلندی",
        "عرض",
        "سایز",
        "روزه",
        "کد",
        "سانت",
        "سانتی",
        "سانتیمتر",
        "سانتی‌متر",
        "cm",
        "gr",
        "پهنا",
        "گرم",
        "رنگ",
        "قدم",
    ]
    irrelevant_pattern = "|".join(map(re.escape, irrelevant_terms))
    irrelevant_regex = rf"({irrelevant_pattern})"
    if re.search(irrelevant_regex, token):
        return True
    return False


def does_token_contains_connector_words(token):
    connector_words = ["و", "تا", "از", "به","ta"]

    pattern = rf"({'|'.join(map(re.escape, connector_words))})"

    if re.match(pattern, token):
        return True
    return False


def tokenize_and_label(text):
    tokens = re.findall(r"\S+|\n", text)
    labels = ["O"] * len(tokens)

    tokens = [clean_token(t) for t in tokens]

    for i, token in enumerate(tokens):
        if not re.search(r"\d", token) or labels[i] != "O":
            continue

        if does_it_contain_price_appendix(token, price_appendix_terms):
            if (
                i > 0
                and tokens[i + 1]
                and is_it_irrelevant_digit(tokens[i - 1], next_token=tokens[i + 1])
            ) or does_token_contain_irrelevant_term(token):
                continue

            if (
                i > 1
                and is_it_irrelevant_digit(tokens[i - 1])
            ) or does_token_contain_irrelevant_term(token):
                continue

            if does_token_contains_connector_words(token):
                continue

            labels[i] = "B-PRICE"
            continue

        if does_it_contain_price_prefix(token, price_prefix_terms):
            if does_token_contain_irrelevant_term(token):
                continue
            if does_token_contains_connector_words(token):
                continue
            labels[i] = "B-PRICE"
            j = i + 1
            if j > len(tokens) - 1:
                continue
            while j < len(tokens) and re.match(r"^\d+$", tokens[j]):
                if (
                    is_pure_digits(tokens[j])
                    or does_it_contain_price_appendix(token, price_appendix_terms)
                ) and not is_it_phone_number(tokens[j]):
                    labels[j] = "I-PRICE"
                j += 1

        before, after = get_neibouring(tokens, i, 2)
        has_prefix = any(term in before for term in price_prefix_terms)
        has_appendix = any(term in after for term in price_appendix_terms)
        if has_prefix or has_appendix:
            if i > 0 and is_it_irrelevant_digit(tokens[i - 1]):
                continue

            if i > 1 and is_it_irrelevant_digit(
                tokens[i - 1]
            ):
                continue

            if does_token_contain_irrelevant_term(tokens[i]):
                continue

            if has_prefix:
                if is_it_phone_number(tokens[i]):
                    continue
                if does_token_contains_connector_words(token):
                    continue
                labels[i] = "B-PRICE"
                j = i + 1

                if j > len(tokens) - 1:
                    continue

                while j < len(tokens) and re.match(r"^\d+$", tokens[j]):
                    if is_it_phone_number(tokens[j]) or is_it_irrelevant_digit(
                        tokens[j - 1]
                    ):
                        j += 1
                        continue
                    elif is_pure_digits(tokens[j]) and not is_it_irrelevant_digit(
                        prev_token=tokens[j - 1]
                    ):
                        labels[j] = "I-PRICE"
                    elif (
                        does_it_contain_price_appendix(token, price_appendix_terms)
                        and not does_token_contain_irrelevant_term(tokens[j - 1])
                        and not does_token_contain_irrelevant_term(tokens[j])
                        and not is_it_irrelevant_digit(prev_token=tokens[j - 1])
                    ):
                        labels[j] = "I-PRICE"
                    j += 1

                    if does_token_contains_connector_words(tokens[j]):
                        continue

            labels[i] = "B-PRICE"

            continue

    i = 0
    while i < len(labels):
        if labels[i] == "B-PRICE":
            if is_pure_digits(tokens[i - 1]):
                if tokens[i - 2] and tokens[i - 2] not in [
                    "و",
                    "تا",
                    "از",
                    "به",
                    "ارسال",
                    "کد"
                ]:
                    labels[i - 1] = "B-PRICE"
                    labels[i] = "I-PRICE"

            start = i
            end = start + 1
            while end < len(labels) and labels[end] in ("B-PRICE", "I-PRICE"):
                end += 1

            for j in range(start + 1, end + 2):
                if j >= len(tokens):
                    break
                token_j = tokens[j]
                if (
                    is_pure_digits(token_j)
                    or token_j in price_appendix_terms
                    or does_it_contain_price_appendix(token_j, price_appendix_terms)
                ) and not is_it_phone_number(token_j):
                    if is_it_irrelevant_digit(
                        tokens[j - 1],
                    ):
                        for k in range(j, end):
                            labels[k] = "O"

                        break

                    labels[j] = "I-PRICE"

            i = end
        else:
            i += 1

    return list(zip(tokens, labels))


def process_dataset(file_path, labeled_output_path, original_output_path):
    data = pd.read_csv(file_path, header=None, names=["text"])
    data["text"] = data["text"].astype(str)

    random_rows = data.sample(n=10)

    random_rows.to_csv(
        original_output_path, index=False, header=False, encoding="utf-8"
    )

    labeled_data = []
    for _, row in random_rows.iterrows():
        text = row["text"]
        tokenized_and_labeled = tokenize_and_label(text)
        labeled_data.extend(tokenized_and_labeled + [("", "")])

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
