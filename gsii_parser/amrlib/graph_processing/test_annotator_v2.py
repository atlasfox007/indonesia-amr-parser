import torch
from transformers import pipeline
from nlp_id.tokenizer import Tokenizer
from nlp_id.postag import PosTag
from nlp_id.lemmatizer import Lemmatizer 

####################################################################
# Indonesia Tokenizer

id_tokenizer = Tokenizer()

input_words_1 = "Otoritas Paris, Belgium, dan Belanda"
input_words = id_tokenizer.tokenize(input_words_1)
print(input_words)

####################################################################
# NER TAGGER

# BERT NER Tagger
# TODO : the ner results parsing logic
ner_pipeline = pipeline("token-classification", model="cahya/bert-base-indonesian-NER")

ner_results = []

for word in input_words:
    ner_res = ner_pipeline(word)

    if(len(ner_res) < 1):
        ner_results.append('O')
    else:
        ner_results.append(ner_res[0]["entity"])
print(ner_results)
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
print(input_words_1)
lemma_result = lemmatizer.lemmatize(input_words_1)
print(lemma_result)
####################################################################