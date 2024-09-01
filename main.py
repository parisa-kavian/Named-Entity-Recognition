
from application.use_cases import train_ner_model

if __name__ == "__main__":
    model, tokenizer = train_ner_model()
    print("Training complete. Model and tokenizer saved.")