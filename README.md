# Named-Entity-Recognition

This project implements a Named Entity Recognition (NER) system using BERT (Bidirectional Encoder Representations from Transformers). NER is a natural language processing (NLP) technique designed to extract and classify important information from text. It focuses on identifying and categorizing named entities, which include key subjects such as names, places, companies, events, products, topics, tenses, monetary values, and percentages.


# Conll2003 dataset

The CoNLL-2003 dataset is used for training, which is a popular benchmark for NER tasks. Provides tagged data for four types of entities:

Persons: Names of people
Locations: Names of places
Organizations: Names of companies or institutions
Miscellaneous entities: Other entities not covered by the above categories

# Project Structure
The project is organized into the following directories:

domain/: contains core data models used in the NER task.

models.py: defines classes for core data models such as NER tokens and tags.
application/: contains use cases for training and evaluating the model.

use_cases.py: implements the training and evaluation workflow for the NER model.
infrastructure/: provides utility functions for data loading, model handling, and metric computation.

data_loader.py: handles loading and processing datasets.
model_utils.py: contains functions for loading and training the model, including saving and loading utilities.
metric_utils.py: includes functions for loading and computing metrics.
presentation/: the entry point for running the training process.

main.py: executes the training process and saves the model and tokenizer.
test.py: designes to evaluate the performance of the trained NER model on a given dataset.
# Requirements.txt 
here’s a requirements.txt file that includes the necessary packages

# Note
If you run NER.ipynb code, please use GPU T4 to avoid errors and increase processing speed.

