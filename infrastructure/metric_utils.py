from datasets import load_metric as load_datasets_metric
import numpy as np

def load_metric():
    try:
        metric = load_datasets_metric("seqeval", trust_remote_code=True)
        return metric
    except Exception as e:
        print(f"Error loading metric: {e}")
        raise

def compute_metrics(eval_preds, label_list, metric):
    try:
        pred_logits, labels = eval_preds
        pred_logits = np.argmax(pred_logits, axis=2)
        predictions = [
            [label_list[pred] for pred, l in zip(prediction, label) if l != -100]
            for prediction, label in zip(pred_logits, labels)
        ]
        true_labels = [
            [label_list[l] for pred, l in zip(prediction, label) if l != -100]
            for prediction, label in zip(pred_logits, labels)
        ]
        results = metric.compute(predictions=predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    except Exception as e:
        print(f"Error computing metrics: {e}")
        raise

metric = load_metric()