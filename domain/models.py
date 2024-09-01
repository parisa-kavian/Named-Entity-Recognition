

from typing import List, Dict

class NERData:
    def __init__(self, tokens: List[str], ner_tags: List[int]):
        self.tokens = tokens
        self.ner_tags = ner_tags

class NERTokenizedInput:
    def __init__(self, input_ids: List[int], attention_mask: List[int], labels: List[int]):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels


