"""
File: chunk_vectorization.py
Author: Nathan Collard <ncollard@openblackbox.be>
Contact: opensource@openblackbox.be
License: MIT License
Project URL: https://github.com/OpenBlackBoxLab/OpenRAG

This file contains the implementation of various vectorizers used for chunk vectorization in the OpenRAG project.
The vectorizers include TF-IDF, Word2Vec, BERT, and ADA vectorizers. These vectorizers are used to convert text chunks into numerical vectors.
The vectorized data is then stored using the Azure Helper module.

Copyright (c) 2024 Open BlackBox

This file is part of OpenRAG and is released under the MIT License.
See the LICENSE file in the root directory of this project for details.
"""

from tqdm import tqdm
from ..utils import azure_helper as azure_handler

# Base class for vectorizers
class Vectorizer:
    def vectorize(self, text):
        raise NotImplementedError("Subclasses must implement this method")

# TF-IDF vectorizer
class TFIDFVectorizer(Vectorizer):
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer()
    
    def vectorize(self, text):
        return self.vectorizer.fit_transform([text]).toarray()[0]

# Word2Vec vectorizer
class Word2VecVectorizer(Vectorizer):
    def __init__(self):
        from gensim.models import Word2Vec
        self.model = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)
    
    def vectorize(self, text):
        # Training on the fly - may not be optimal
        self.model.build_vocab([text.split()], update=True)
        self.model.train([text.split()], total_examples=1, epochs=self.model.epochs)
        return self.model.wv.get_mean_vector(text.split())

# BERT vectorizer
class BERTVectorizer(Vectorizer):
    def __init__(self):
        from transformers import BertTokenizer, BertModel
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
    
    def vectorize(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()[0].tolist()

# ADA vectorizer
class ADAVectorizer(Vectorizer):
    def __init__(self):
        from openai import OpenAI
        self.openai = OpenAI()
    
    def vectorize(self, text):
        response = self.openai.embeddings.create(
            input=text,
            model="text-embedding-3-large",
            encoding_format="float"
        )
        return response.data[0].embedding

def get_vectorizer(vectorizer_type):
    vectorizers = {
        'tfidf': TFIDFVectorizer,
        'word2vec': Word2VecVectorizer,
        'bert': BERTVectorizer,
        'ada': ADAVectorizer
    }
    return vectorizers[vectorizer_type]()

def pad_vector(vector, target_dim):
    padding_length = target_dim - len(vector)
    if padding_length > 0:
        vector.extend([0.0] * padding_length)
    return vector

def vectorize_and_store(file_name, vectorizer_type, expected_dim):
    chunks = azure_handler.get_chunked_dict(file_name)
    vectorizer = get_vectorizer(vectorizer_type)
    vector_data = []

    for chunk in tqdm(chunks.values()):
        text = chunk['text']
        vector = vectorizer.vectorize(text)
        padded_vector = pad_vector(vector, expected_dim)
        vector_data.append(padded_vector)
    
    azure_handler.put_vectorized_dict(file_name, vector_data)
