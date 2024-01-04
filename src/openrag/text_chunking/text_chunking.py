"""
File: text_chunking.py
Author: Nathan Collard <ncollard@openblackbox.be>
Contact: opensource@openblackbox.be
License: MIT License
Project URL: https://github.com/OpenBlackBoxLab/OpenRAG

The purpose of this file is to provide functionality for text chunking, which involves dividing a text into smaller chunks or segments. 
The code includes functions for calculating the number of tokens in a string based on a specified encoding, 
as well as determining the overlap of text chunks. It also imports modules such as spacy and azure_helper from the project's utils package.

Copyright (c) 2024 Open BlackBox

This file is part of OpenRAG and is released under the MIT License.
See the LICENSE file in the root directory of this project for details.
"""
import spacy
from ..utils import azure_helper as azure_handler
from tiktoken import get_encoding

# Configuration Constants
OVERLAP_SIZE_SENTENCES = 1 
OVERLAP_SIZE_TOKENS = 40
CHUNK_SIZE_SENTENCES = 4
CHUNK_SIZE_TOKENS_MIN = 126 + OVERLAP_SIZE_TOKENS
CHUNK_SIZE_TOKENS_MAX = 256

def num_tokens_in_string(string, encoding_name="cl100k_base"):
    """
    Calculate the number of tokens in a string based on a specified encoding.

    Args:
        string (str): The text string to encode.
        encoding_name (str): The name of the encoding to use.

    Returns:
        int: The number of tokens in the string.
    """
    encoding = get_encoding(encoding_name)
    return len(encoding.encode(string))

def get_overlap(chunks, overlap_size):
    """
    Determine the overlap of chunks, cutting at the nearest word boundary.

    Args:
        chunks (list): A list of text chunks.
        overlap_size (int): The size of the overlap in characters.

    Returns:
        tuple: A list with a single overlap chunk, and the number of tokens in the overlap.
    """
    fluent_text = ' '.join(chunks)
    char_index = len(fluent_text) - overlap_size * 3
    while char_index > 0 and fluent_text[char_index] != ' ':
        char_index -= 1

    overlap = fluent_text[char_index:].strip()
    num_tokens_overlap = num_tokens_in_string(overlap)

    return [overlap], num_tokens_overlap

def load_spacy_models():
    """
    Load and return French and Dutch spaCy models.

    Returns:
        tuple: Loaded French and Dutch spaCy models.
    """
    nlp_fr = spacy.load("fr_dep_news_trf")  # French model
    nlp_nl = spacy.load("nl_core_news_lg")  # Dutch model
    return nlp_fr, nlp_nl

def split_into_sentences(text, nlp):
    """
    Split text into sentences using a spaCy model.

    Args:
        text (str): The text to split.
        nlp: The spaCy language model.

    Returns:
        list: A list of sentence strings.
    """
    return [sent.text.strip() for sent in nlp(text).sents]

def chunk_sentences(sentences, min_chunk_size, max_chunk_size, overlap_size):
    """
    Chunk sentences into specified sizes, considering token count.

    Args:
        sentences (list): A list of sentence dictionaries.
        min_chunk_size (int): The minimum size for a chunk.
        max_chunk_size (int): The maximum size for a chunk.
        overlap_size (int): The size of overlap between chunks.

    Returns:
        dict: A dictionary with chunked sentences.
    """
    chunks = []
    current_chunk = []
    token_count = 0

    for sentence in sentences:
        num_tokens = num_tokens_in_string(sentence['text'])
        current_chunk.append(sentence['text'])
        token_count += num_tokens

        if token_count >= min_chunk_size:
            if token_count <= max_chunk_size:
                chunks.append(current_chunk[:])
                current_chunk, token_count = get_overlap(current_chunk, overlap_size)
            else:
                # Split sentence if chunk size exceeds max limit
                split_sentence = sentence['text'].split()
                half_index = len(split_sentence) // 2
                first_half = ' '.join(split_sentence[:half_index])
                second_half = ' '.join(split_sentence[half_index:])

                current_chunk[-1] = first_half
                chunks.append(current_chunk[:])
                current_chunk, token_count = get_overlap(current_chunk, overlap_size)

                current_chunk.append(second_half)
                token_count += num_tokens_in_string(second_half)

    if current_chunk and token_count > overlap_size:
        chunks.append(current_chunk)

    return {'chunks': [' '.join(chunk) for chunk in chunks]}

def overlapping_chunking(pages, min_chunk_size, max_chunk_size, overlap_size):
    """
    Chunk text from pages into overlapping chunks based on token count.

    Args:
        pages (list): A list of pages where each page contains text and its page number.
        min_chunk_size (int): The minimum size of each chunk in tokens.
        max_chunk_size (int): The maximum size of each chunk in tokens.
        overlap_size (int): The desired overlap between adjacent chunks in tokens.

    Returns:
        dict: A dictionary of chunked sentences with associated metadata.
    """
    print("Loading spaCy models...")
    nlp_fr, nlp_nl = load_spacy_models()  # Load spacy models

    print("Splitting into sentences...")
    all_sentences = []
    for page_text, page_num in pages:
        sentences = split_into_sentences(page_text, nlp_fr)
        for sentence in sentences:
            all_sentences.append({
                'text': sentence,
                'page': page_num
            })

    return chunk_sentences(all_sentences, min_chunk_size, max_chunk_size, overlap_size)

def chunk_and_save(file_name):
    """
    Chunk a text file and save the chunks to Azure Blob Storage.

    Args:
        file_name (str): The name of the file to be chunked and saved.

    Returns:
        None
    """
    print("Chunking and saving " + file_name)
    data = azure_handler.get_extracted_dict(file_name)
    print("Loaded " + file_name)
    pages = [(entry["text"], entry["page"]) for entry in data]
    print("Chunking " + file_name)
    chunks = overlapping_chunking(pages, CHUNK_SIZE_TOKENS_MIN, CHUNK_SIZE_TOKENS_MAX, OVERLAP_SIZE_TOKENS)
    print("Saving " + file_name)
    azure_handler.put_chunked_dict(file_name, chunks)