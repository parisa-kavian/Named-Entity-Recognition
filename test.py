
from transformers import pipeline, AutoModelForTokenClassification, BertTokenizerFast

try:

    model = AutoModelForTokenClassification.from_pretrained("ner_model")
    tokenizer = BertTokenizerFast.from_pretrained("tokenizer")
    

    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
    

    test_sentence = "Apple Inc. is a technology company based in Cupertino, California. Tim Cook is the CEO of Apple. The company was founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne. Apple's headquarters are located in the United States."
    
 
    ner_results = ner_pipeline(test_sentence)
    

    for entity in ner_results:
        print(f"Word: {entity['word']}, Entity: {entity['entity']}, Confidence: {entity['score']:.4f}")
except Exception as e:
    print(f"Error during inference: {e}")
    raise
