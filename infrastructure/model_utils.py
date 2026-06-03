# infrastructure/model_utils.py
from transformers import (
    BertTokenizerFast,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from infrastructure.metric_utils import load_metric, compute_metrics


def get_tokenizer():
    return BertTokenizerFast.from_pretrained("bert-base-uncased")


def get_model(label_list):
    id2label = {i: l for i, l in enumerate(label_list)}
    label2id = {l: i for i, l in enumerate(label_list)}
    return AutoModelForTokenClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )


def build_trainer(model, tokenizer, tokenized_datasets, label_list):
    args = TrainingArguments(
        output_dir="ner-bert-conll2003",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
    )
    data_collator = DataCollatorForTokenClassification(tokenizer)
    metric = load_metric()

    return Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, label_list, metric),
    )


def save_model_and_tokenizer(model, tokenizer, output_dir="ner_model"):
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
