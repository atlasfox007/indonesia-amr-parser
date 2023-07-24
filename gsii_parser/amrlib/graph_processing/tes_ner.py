from transformers import pipeline

# Load the NER pipeline for Bahasa Indonesia
ner_pipeline = pipeline("token-classification", model="cahya/bert-base-indonesian-NER")

# Example usage
text = "Paris"
ner_results = ner_pipeline(text)

for result in ner_results:
    print(result)