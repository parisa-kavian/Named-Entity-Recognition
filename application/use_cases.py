
from infrastructure.data_loader import load_dataset, tokenize_and_align_labels
from infrastructure.model_utils import get_tokenizer, get_model, train_model, save_model_and_tokenizer
from infrastructure.metric_utils import load_metric, compute_metrics

def train_ner_model():

    data = load_dataset()


    tokenizer = get_tokenizer()


    tokenized_datasets = data.map(tokenize_and_align_labels, batched=True)


    model = get_model()

  
    trainer = train_model(model, tokenizer, tokenized_datasets)

 
    trainer.train()

    save_model_and_tokenizer(model, tokenizer)

    return model, tokenizer

