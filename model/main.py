import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

local_directory = "./../saved_model"
tokenizer = AutoTokenizer.from_pretrained(local_directory)
model = AutoModelForTokenClassification.from_pretrained(local_directory)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("Model runs os:", device)

def extract_price(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    predicted_labels = predictions[0].cpu().numpy()
    
    extracted_price = []
    for token, label in zip(tokens, predicted_labels):
        if label == 1:  
            extracted_price.append(token)
    
    price = tokenizer.convert_tokens_to_string(extracted_price)
    return price

file_path = "./../data/post_data/extracted_data_full_cleaned_persianed_digits.txt"
data = pd.read_csv(file_path, header=None, names=["sentence"])

random_sentences = data.sample(n=100)

results = []
for sentence in random_sentences["sentence"]:
    extracted_price = extract_price(sentence)
    results.append({"sentence": sentence, "extracted_price": extracted_price})

output_file = "extracted_prices.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for result in results:
        f.write(result["sentence"] + "\n")  
        f.write(":::::::::::::::::::::::::" + str(result["extracted_price"]) + "\n") 
        f.write("\n" + "-" * 50 + "\n\n")  

print(f"Extracted prices saved to {output_file}")
