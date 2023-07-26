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


ner_pipeline_result = ner_pipeline(input_words_1)
ner_dict = {}


# Turn the ner_pipeline_result to dictionary
for i in range(len(ner_pipeline_result)):
    ner_dict[ner_pipeline_result[i]['word']] = ner_pipeline_result[i]['entity']

ner_result = []
for word in input_words:
    word_check = word.lower()
    if(word_check in ner_dict):
        ner_result.append(ner_dict[word_check])
    else:
        ner_result.append('O') # Not present in ner_pipeline_result

print(ner_result)

# for word in input_words:
#     ner_res = ner_pipeline(word)

#     if(len(ner_res) < 1):
#         ner_results.append('O')
#     else:
#         ner_results.append(ner_res[0]["entity"])
# print(ner_results)
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
lemma_result = []

for word in input_words:
    lm = lemmatizer.lemmatize(word)
    if(len(lm) < 1):
        lemma_result.append(word)
    else:
        lemma_result.append(lm)
print(lemma_result)
####################################################################