import torch
from transformers import BertTokenizer, BertModel

# Load the pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Input text
input_text = "This is a sample input text to examine embeddings."

# Tokenize the input text and convert it to input tensors
inputs = tokenizer(input_text, return_tensors="pt")

# Pass the input tensors through the BERT model to get outputs
with torch.no_grad():
    outputs = model(**inputs)

# Extract the token embeddings and positional embeddings from the model outputs
token_embeddings = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)
positional_embeddings = model.embeddings.position_embeddings.weight  # Shape: (sequence_length, hidden_size)

# Convert the input text to tokens (including special tokens) and decode them
tokens = tokenizer.tokenize(input_text)
decoded_tokens = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens))

# Print the tokens and their corresponding embeddings
for i, (token, token_embedding) in enumerate(zip(decoded_tokens, token_embeddings[0])):
    print(f"Token: {token}")
    print(f"Token Embedding: {token_embedding}")
    print("--------")

# Print the positional embeddings
print("Positional Embeddings:")
print(positional_embeddings)