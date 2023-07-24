import torch
from transformers import BertTokenizer, BertForTokenClassification
from nlp_id.tokenizer import Tokenizer
from nlp_id.postag import PosTag
from nlp_id.lemmatizer import Lemmatizer 

####################################################################
# Indonesia Tokenizer

id_tokenizer = Tokenizer()

input_words = "Presiden Indonesia yang terakhir adalah Joko Widodo !"
input_words = id_tokenizer.tokenize(input_words)
print(input_words)

####################################################################
# NER TAGGER

# BERT NER Tagger
ner_tokenizer = BertTokenizer.from_pretrained("cahya/bert-base-indonesian-NER")
ner = BertForTokenClassification.from_pretrained("cahya/bert-base-indonesian-NER")

# Input sentence (array of words)

# Tokenize the input sentence and get attention_mask
tokenized_input = ner_tokenizer(" ".join(input_words), add_special_tokens=False, return_tensors="pt", padding=True, truncation=True, max_length=512)
input_ids = tokenized_input["input_ids"]
attention_mask = tokenized_input["attention_mask"]

# Predict the NER tags
with torch.no_grad():
    predictions = ner(input_ids, attention_mask=attention_mask)
logits = predictions[0]

active_logits = logits[attention_mask == 1] # Consider only non-padding tokens
flattened_predictions = torch.argmax(active_logits, axis=1)
id2label = ner.config.id2label

ner_token_predictions =[id2label[i] for i in flattened_predictions.numpy()]
print(ner_token_predictions)
####################################################################

####################################################################
# POS TAGGER
postagger = PosTag()
pos_tag = postagger.get_pos_tag(" ".join(input_words))
pos_tag_result = [x[1] for x in pos_tag]
print(pos_tag_result)
####################################################################
# LEMMATIZER
lemmatizer = Lemmatizer() 
lemma_result = lemmatizer.lemmatize(" ".join(input_words))
print(lemma_result.split())
####################################################################