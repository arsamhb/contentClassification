import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from sklearn.model_selection import train_test_split

# Step 1: Load Your Data
# CONSIDER THIS DATA SHOULD HAVE A TEXT AND A LABELS I THINK IT WOULD BE MUCH SIMPLER TO USE THEIR HASHTAGS THEY SHOULD
# BE GOOD ENOUGH IF THEY WERE NOT WE START TO USE HANDY LABELING
df = pd.read_csv("train_data.csv")  

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Just converting to list as the pipelines of data work better in this way for py 
train_texts = train_df["text"].tolist()
train_labels = train_df["label"].tolist()

val_texts = val_df["text"].tolist()
val_labels = val_df["label"].tolist()

# Step 2: Prepare the Tokenizer and Encode the Data

# We pick a pretrained model name, for instance "bert-base-uncased".
# Note: If working with a language other than English, choose an appropriate model.
model_name = "bert-base-uncased"

# Load the tokenizer associated with the chosen model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# The tokenizer transforms text into a list of integers (token IDs).
# We'll define a function to tokenize our texts.
def tokenize_function(texts):
    # The tokenizer automatically handles splitting into tokens, adding special tokens [CLS], [SEP],
    # and returns a dictionary with 'input_ids' and 'attention_mask'.
    return tokenizer(texts, truncation=True, padding=True, max_length=128)

# Tokenize the training and validation sets
train_encodings = tokenize_function(train_texts)
val_encodings = tokenize_function(val_texts)

############################################################
# Explanation of the tokenization section:
#
# model_name = "bert-base-uncased":
#  - A string identifier for the pretrained BERT model from the Hugging Face model hub.
#
# AutoTokenizer.from_pretrained(model_name):
#  - Downloads and initializes a tokenizer that corresponds to "bert-base-uncased".
#
# tokenize_function:
#  - Defines a helper function that applies the tokenizer to a list of texts.
#  - `truncation=True`: If the text is longer than the model's maximum sequence length (128 here),
#    it will be truncated.
#  - `padding=True`: Pad the sequences so they all have the same length.
#
# train_encodings and val_encodings:
#  - Dictionaries containing "input_ids" (the token IDs), "attention_mask" (masks to ignore padding),
#    and possibly "token_type_ids".
############################################################

############################################################
# Step 3: Convert Labels to Numeric Indices
############################################################

# Suppose labels are strings like "bag", "scarf", "watch".
# We need to map them to numeric IDs for classification.
# Let's build a label-to-index mapping:
label_list = sorted(list(set(train_labels + val_labels)))
label_to_id = {label: i for i, label in enumerate(label_list)}

# Convert the string labels in train/val sets to their numeric IDs
train_label_ids = [label_to_id[l] for l in train_labels]
val_label_ids = [label_to_id[l] for l in val_labels]

############################################################
# Explanation of label mapping:
#
# We collect all unique labels using `set`, convert to list, and sort them to have a stable order.
# label_to_id = {label: i ...} creates a dictionary mapping each label string to an integer index.
# train_label_ids and val_label_ids are now lists of integers instead of strings.
#
# This is necessary because the model outputs logits for each class index, and the loss function
# expects integer class indices.
############################################################

############################################################
# Step 4: Create a Dataset Class for PyTorch
############################################################

class CaptionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        # Number of items in the dataset
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Return a single item of the dataset
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])  # classification labels
        return item

train_dataset = CaptionDataset(train_encodings, train_label_ids)
val_dataset = CaptionDataset(val_encodings, val_label_ids)

############################################################
# Explanation of the dataset class:
#
# - PyTorch expects datasets that can return samples by index.
# - __init__(self, encodings, labels): Store the tokenized inputs (encodings) and their labels.
# - __len__(self): Returns how many samples are in the dataset.
# - __getitem__(self, idx): Returns the tokenized sample and its label at the index idx.
#
# item = {key: torch.tensor(val[idx]) ...}: For each field in encodings (like input_ids, attention_mask),
# we take the idx-th element and turn it into a PyTorch tensor.
# item["labels"] = torch.tensor(self.labels[idx]) sets the label for that item.
############################################################

############################################################
# Step 5: Load the Pretrained Model
############################################################

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_list)  # specify the number of classes
)

############################################################
# Explanation of model loading:
#
# AutoModelForSequenceClassification.from_pretrained:
#  - Downloads the pretrained BERT weights.
#  - Adds a classification head (a linear layer) on top of the [CLS] token representation.
#  - num_labels = number of distinct classes in our dataset.
#
# Under the hood:
#  - The model forward pass will produce a vector of logits (real-valued scores) for each label.
#  - During training, the Trainer will use these logits to compute the cross-entropy loss.
############################################################

############################################################
# Step 6: Define Training Arguments
############################################################

training_args = TrainingArguments(
    output_dir="./results",         # directory where model predictions and checkpoints are saved
    evaluation_strategy="steps",    # evaluate every X steps
    eval_steps=50,                  # evaluate every 50 steps
    per_device_train_batch_size=8,  # batch size for training
    per_device_eval_batch_size=8,   # batch size for evaluation
    num_train_epochs=3,             # number of epochs to train
    logging_steps=50,               # log training info every 50 steps
    save_steps=100,                 # save a checkpoint every 100 steps
    load_best_model_at_end=True,    # load the best model found during training at the end
    metric_for_best_model="accuracy",  # metric to determine best model
    greater_is_better=True
)

############################################################
# Explanation of training arguments:
#
# output_dir="./results":
#  - Directory for saving model checkpoints (weights) and logs.
#
# evaluation_strategy="steps":
#  - Evaluate model at regular intervals (every eval_steps).
#
# per_device_train_batch_size=8 and per_device_eval_batch_size=8:
#  - How many samples per batch during training and evaluation.
#    Batches are processed in parallel on GPU if available.
#
# num_train_epochs=3:
#  - The entire training dataset will be processed 3 times.
#
# logging_steps=50:
#  - Print training progress every 50 steps.
#
# load_best_model_at_end=True:
#  - After training, load the checkpoint that had the best evaluation metric.
############################################################

############################################################
# Step 7: Define a Compute Metrics Function (Optional)
############################################################

from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

############################################################
# Explanation of compute_metrics:
#
# eval_pred is a tuple (logits, labels) returned by the Trainer during evaluation.
# logits: The output of the model before the softmax.
# labels: The true labels.
#
# predictions = logits.argmax(axis=-1):
#  - Finds the index of the largest logit for each sample, which corresponds to the predicted class.
#
# accuracy_score(labels, predictions):
#  - Fraction of correctly predicted labels.
############################################################

############################################################
# Step 8: Initialize the Trainer and Train the Model
############################################################

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Start training
trainer.train()

############################################################
# Explanation of the Trainer:
#
# Trainer is a high-level API that:
# - Handles the training loop.
# - Feeds batches of data through the model.
# - Computes the loss, does backpropagation, and updates model weights.
# - Evaluates periodically if asked.
#
# trainer.train():
# - Runs the training loop for the specified number of epochs.
############################################################

############################################################
# After training, the best model is loaded, and we can use it for inference:
############################################################

# Let's say we have some new unlabeled text data:
new_texts = ["این کیف دخترانه جدید هست؟", "یک شال پشمی برای زمستان میخواهم"]

# Tokenize new texts
new_encodings = tokenizer(new_texts, truncation=True, padding=True, return_tensors="pt")

# Predict
outputs = model(**new_encodings)
# outputs is of type SequenceClassifierOutput, containing logits.

# Get predicted class indices
predicted_class_indices = torch.argmax(outputs.logits, dim=1).tolist()

# Convert indices back to label names
predicted_labels = [label_list[i] for i in predicted_class_indices]

print(predicted_labels)

############################################################
# Explanation of inference:
#
# new_texts: A list of new captions without labels.
#
# tokenizer(..., return_tensors="pt"):
#  - Tokenize and return PyTorch tensors directly.
#
# outputs = model(**new_encodings):
#  - Forward pass: Input token IDs and attention masks through the model.
#  - outputs.logits: A tensor of shape (batch_size, num_labels) containing the classification scores.
#
# torch.argmax(outputs.logits, dim=1):
#  - Finds the label with the highest score for each sample.
#
# predicted_labels:
#  - Convert numeric predictions back into original label strings.
############################################################

---

**What just happened mathematically inside the model?**  
- Input text is converted to token IDs.  
- The BERT model computes hidden states through multiple self-attention layers. Self-attention uses a weighted combination of token embeddings:
  \[
  \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
  \]
  where \(Q, K, V\) are query, key, and value matrices derived from the input embeddings, and \(d_k\) is the dimension of the keys. The `softmax` ensures weights sum to 1, focusing attention on certain tokens.  
- After several layers, the [CLS] token’s representation is passed to a linear layer for classification:
  \[
  \text{logits} = X_{[CLS]}W + b
  \]
  where \(X_{[CLS]}\) is the final hidden state of the [CLS] token, and \(W, b\) are the linear layer’s parameters.  
- The training uses cross-entropy loss to adjust \(W\) and \(b\), as well as all the BERT parameters, to minimize the difference between predicted and true labels.  
- On inference, the `argmax` of `logits` selects the predicted class.

This completes **Step 1**: We have shown how to fine-tune a pretrained BERT model using a small labeled dataset and then use it to label new captions automatically.
