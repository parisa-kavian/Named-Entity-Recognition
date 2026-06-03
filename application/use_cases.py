from infrastructure.data_loader import load_dataset, tokenize_and_align_labels
from infrastructure.model_utils import get_tokenizer, get_model, build_trainer, save_model_and_tokenizer


def train_ner_model():
    data = load_dataset()
    label_list = data["train"].features["ner_tags"].feature.names

    tokenizer = get_tokenizer()
    tokenized_datasets = data.map(tokenize_and_align_labels, batched=True)

    model = get_model(label_list)
    trainer = build_trainer(model, tokenizer, tokenized_datasets, label_list)
    trainer.train()

    save_model_and_tokenizer(model, tokenizer)
    return model, tokenizer
