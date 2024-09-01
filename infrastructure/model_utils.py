# infrastructure/model_utils.py

from transformers import BertTokenizerFast, AutoModelForTokenClassification, TrainingArguments, Trainer
from infrastructure.metric_utils import load_metric, compute_metrics
from transformers import DataCollatorForTokenClassification

def get_tokenizer():
    try:
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise

def get_model():
    try:
        model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=9)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def train_model(model, tokenizer, tokenized_datasets):
    args = TrainingArguments(
        "test-ner",
        evaluation_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
    )
    data_collator = DataCollatorForTokenClassification(tokenizer)
    metric = load_metric()
    label_list = tokenized_datasets["train"].features["ner_tags"].feature.names

    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, label_list, metric)
    )
    return trainer

def save_model_and_tokenizer(model, tokenizer):
    try:
        model.save_pretrained("ner_model")
        tokenizer.save_pretrained("tokenizer")
    except Exception as e:
        print(f"Error saving model or tokenizer: {e}")
        raise

