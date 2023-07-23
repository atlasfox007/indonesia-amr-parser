import torch
from transformers import BertTokenizer, BertForTokenClassification

# Indonesia Tokenizer
from nlp_id.tokenizer import Tokenizer 
id_tokenizer = Tokenizer() 

# BERT Tokenizer and NER Tagger
tokenizer = BertTokenizer.from_pretrained("cahya/bert-base-indonesian-NER")
ner = BertForTokenClassification.from_pretrained("cahya/bert-base-indonesian-NER")

# Input sentence (array of words)
input_words = "Presiden Indonesia yang terakhir adalah Joko Widodo !"
input_words = id_tokenizer.tokenize(input_words)

print(input_words)
# Tokenize the input sentence and get attention_mask
tokenized_input = tokenizer(" ".join(input_words), add_special_tokens=False, return_tensors="pt", padding=True, truncation=True, max_length=512)
input_ids = tokenized_input["input_ids"]
attention_mask = tokenized_input["attention_mask"]

# Predict the NER tags
with torch.no_grad():
    predictions = ner(input_ids, attention_mask=attention_mask)
logits = predictions[0]

active_logits = logits[attention_mask == 1] # Consider only non-padding tokens
flattened_predictions = torch.argmax(active_logits, axis=1)
id2label = ner.config.id2label

token_predictions =[id2label[i] for i in flattened_predictions.numpy()]
print(input_words)
print(token_predictions)
