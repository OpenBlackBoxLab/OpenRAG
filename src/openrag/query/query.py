"""
File: query.py
Author: Nathan Collard <ncollard@openblackbox.be>
Contact: opensource@openblackbox.be
License: MIT License
Project URL: https://github.com/OpenBlackBoxLab/OpenRAG

This file, query.py, appears to be a part of a project called OpenRAG. It contains functions related to querying and filtering results. Specifically, 
it includes a function called vectorize_question that vectorizes a given question text using the ADA vectorizer, and a function called filter_non_adjacent_indices 
that filters out non-adjacent indices from a list of results based on a maximum number of neighbors. The purpose of this file seems to be to provide functionality 
for querying and filtering data in the OpenRAG project.

Copyright (c) 2024 Open BlackBox

This file is part of OpenRAG and is released under the MIT License.
See the LICENSE file in the root directory of this project for details.
"""
from ..chunk_vectorization import chunk_vectorization as vectorize
from ..utils import azure_helper as azure_helper
import json

def vectorize_question(question_text):
    """
    Vectorize a given question text using the ADA vectorizer.

    Args:
        question_text (str): The text of the question to vectorize.

    Returns:
        list: The vector representation of the question.
    """
    vectorizer = vectorize.get_vectorizer("ada")
    return vectorizer.vectorize(question_text)

def filter_non_adjacent_indices(results, max_neighbors):
    """
    Filter out non-adjacent indices from the results.

    Args:
        results (list): A list of result objects with 'id' and 'distance' attributes.
        max_neighbors (int): The maximum number of neighbors to return.

    Returns:
        list: Filtered list of indices and their distances.
    """
    final_indices = []

    for result in results:
        if result.id not in [idx[0] for idx in final_indices]:
            final_indices.append([result.id, result.distance])
            if len(final_indices) >= max_neighbors:
                break

            # Handle adjacent indices
            for offset in [-1, 1]:
                adjacent_index = result.id + offset
                if adjacent_index >= 0 and adjacent_index not in [idx[0] for idx in final_indices]:
                    final_indices.append([adjacent_index, ""])
                    if len(final_indices) >= max_neighbors:
                        break

    return final_indices

def find_text_chunks(chunk_id, file_path):
    """
    Find text chunks based on their chunk ID.

    Args:
        chunk_id (int): The ID of the chunk to find.
        file_path (str): The path to the file containing chunk metadata.

    Returns:
        dict: The text chunk if found, otherwise None.
    """
    with open(file_path, 'r') as file:
        data_dict = json.load(file)

    for key, value in data_dict.items():
        if value["start"] <= chunk_id <= value["end"]:
            index_in_file = chunk_id - value["start"]
            data_dict_file = azure_helper.getChunkedDict(key)
            return data_dict_file.get(f"chunk_{index_in_file}")

    return None
