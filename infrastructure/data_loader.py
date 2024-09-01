
from infrastructure.model_utils import get_tokenizer
from datasets import load_dataset as load_hf_dataset
from transformers import BertTokenizerFast

def load_dataset():
    try:
        data = load_hf_dataset("eriktks/conll2003")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def tokenize_and_align_labels(examples, label_all_tokens=True):
    tokenizer = get_tokenizer()
    try:
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    except Exception as e:
        print(f"Error during tokenization and alignment: {e}")
        raise
