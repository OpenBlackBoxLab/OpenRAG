"""
File: text_chunking.py
Author: Nathan Collard <ncollard@openblackbox.be>
Contact: opensource@openblackbox.be
License: MIT License
Project URL: https://github.com/OpenBlackBoxLab/OpenRAG

The purpose of this file is to provide functionality for text chunking, which involves dividing a text into smaller chunks or segments. 
The code includes functions for calculating the number of tokens in a string based on a specified encoding, 
as well as determining the overlap of text chunks.

Copyright (c) 2024 Open BlackBox

This file is part of OpenRAG and is released under the MIT License.
See the LICENSE file in the root directory of this project for details.
"""
import re
from tqdm import tqdm
from ..utils import azure_storage_handler as azure_handler
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
        sentence_text = sentence['text']
        sentence_page = sentence['page']
        sentence_num = sentence['sentence_num']
        sentences_page = sentence['sentences_page']
        num_tokens = num_tokens_in_string(sentence_text, "cl100k_base")
        current_chunk.append(sentence_text)
        token_count += num_tokens

        # Check if the current chunk (minus the overlap) meets or exceeds the desired size
        if min_chunk_size <= token_count <= max_chunk_size:
            current_chunk.append(sentence_text)
            chunks.append([current_chunk, sentence_page, sentence_num, sentences_page])
            current_chunk, token_count = get_overlap(current_chunk, overlap_size)
        elif token_count > max_chunk_size:
            # Split the sentence in two
            words = len(sentence_text)
            sentence_first_part = sentence_text[:-words//2]
            sentence_second_part = sentence_text[-words//2:]
            # Add the first part of the sentence to the current chunk
            current_chunk.append(sentence_first_part) 
            chunks.append([current_chunk, sentence_page, sentence_num, sentences_page])
            # Get started with the new chunk
            current_chunk, token_count = get_overlap(current_chunk, overlap_size)
            current_chunk.append(sentence_second_part)
            num_tokens_new_part = num_tokens_in_string(' '.join(current_chunk), "cl100k_base")
            token_count += num_tokens_new_part

    chunks_dict = {f"chunk_{i}": {"text": " ".join(chunk[0]), "page": chunk[1], "sentence_num": chunk[2], "sentences_page": chunk[3]} 
               for i, chunk in enumerate(chunks, start=1)}
    return chunks_dict

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

    all_sentences = []
    texts = [page_text for page_text, _ in pages]
    pages_info = [(page_num, len(text)) for text, page_num in pages]

    for doc, (page_num, text_length) in tqdm(zip(texts, pages_info), total=len(pages), desc="Chunking"):
        sentences = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)(?<!\w[!?])\s', doc)
        sentences_page = len(sentences)
        for sentence_num, sent in enumerate(sentences, start=1):
            sentence_info = {
                "text": sent.strip(),
                "page": page_num,
                "sentence_num": sentence_num,
                "sentences_page": sentences_page
            }
            all_sentences.append(sentence_info)

    return chunk_sentences(all_sentences, min_chunk_size, max_chunk_size, overlap_size)

def chunk_and_save(file_name):
    """
    Chunk a text file and save the chunks to Azure Blob Storage.

    Args:
        file_name (str): The name of the file to be chunked and saved.

    Returns:
        None
    """
    data = azure_handler.get_extracted_dict(file_name)
    pages = [(entry["text"], entry["page"]) for entry in data]
    chunks = overlapping_chunking(pages, CHUNK_SIZE_TOKENS_MIN, CHUNK_SIZE_TOKENS_MAX, OVERLAP_SIZE_TOKENS)
    azure_handler.put_chunked_dict(file_name, chunks)