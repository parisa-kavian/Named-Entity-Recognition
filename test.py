from transformers import pipeline

ner_pipeline = pipeline(
    "ner", model="ner_model", tokenizer="ner_model", aggregation_strategy="simple"
)

test_sentence = (
    "Apple Inc. is a technology company based in Cupertino, California. "
    "Tim Cook is the CEO of Apple. The company was founded in 1976 by "
    "Steve Jobs, Steve Wozniak, and Ronald Wayne."
)

for entity in ner_pipeline(test_sentence):
    print(f"Word: {entity['word']}, Entity: {entity['entity_group']}, Confidence: {entity['score']:.4f}")
