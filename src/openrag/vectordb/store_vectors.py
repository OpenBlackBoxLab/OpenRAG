"""
File: store_vectors.py
Author: Nathan Collard <ncollard@openblackbox.be>
Contact: opensource@openblackbox.be
License: MIT License
Project URL: https://github.com/OpenBlackBoxLab/OpenRAG

This file contains functions for storing vectors in a Milvus collection.
It provides functions to check the connection status, drop a collection if it exists,
create a collection schema, and store vectors in the collection.

Copyright (c) 2024 Open BlackBox

This file is part of OpenRAG and is released under the MIT License.
See the LICENSE file in the root directory of this project for details.
"""

from pymilvus import (
    connections, utility, Collection, CollectionSchema
)
import time

def check_ping_status(alias):
    """
    Check the connection status to the Milvus database.

    Args:
        alias (str): The alias of the connection.

    Returns:
        None
    """
    if alias in connections.list_connections():
        status = connections.get_connection(alias).ping()
        print(f"Ping status: {status}")
    else:
        print(f"No connection found for alias: {alias}")

def check_and_drop_collection(collection_name):
    """
    Check if a collection exists and drop it if it does.

    Args:
        collection_name (str): The name of the collection to check.

    Returns:
        bool: True if the collection existed and was dropped, False otherwise.
    """
    if utility.has_collection(collection_name):
        print(f"Collection {collection_name} already exists and will be dropped.")
        utility.drop_collection(collection_name)
        print(f"Collection {collection_name} has been dropped.")
        return True
    else:
        print(f"Collection {collection_name} does not exist.")
        return False

def create_collection_schema(fields):
    """
    Create a schema for a Milvus collection.

    Args:
        fields (list): A list of FieldSchema objects.

    Returns:
        CollectionSchema: The collection schema.
    """
    schema = CollectionSchema(fields=fields, description="Collection of text embeddings")
    return schema

def store_vectors(collection_name, schema, vectors, vector_field, sources):
    """
    Store vectors in a Milvus collection.

    Args:
        collection_name (str): The name of the collection.
        schema (CollectionSchema): The schema of the collection.
        vectors (list): List of vectors to store.
        vector_field (str): The field name of vectors in the collection.
        sources (list): List of sources corresponding to each vector.

    Returns:
        None
    """
    # Create or get the collection
    collection = Collection(name=collection_name, schema=schema if not utility.has_collection(collection_name) else None)

    # Insert data in chunks
    chunk_size = 5000
    num_chunks = len(vectors) // chunk_size + (1 if len(vectors) % chunk_size else 0)

    for i in range(num_chunks):
        start_idx, end_idx = i * chunk_size, (i + 1) * chunk_size
        data_chunk = [list(range(start_idx, end_idx)), vectors[start_idx:end_idx], sources[start_idx:end_idx]]
        collection.insert(data_chunk)

    # Flush collection
    _flush_collection(collection)

    # Build index and load collection
    _build_index_and_load(collection, vector_field)

def _flush_collection(collection):
    """
    Flush the collection to write data to disk.

    Args:
        collection (Collection): The collection to flush.

    Returns:
        None
    """
    print("Flushing collection...")
    start_time = time.time()
    collection.flush()
    print(f"Flush completed in {round(time.time() - start_time, 4)} seconds.")

def _build_index_and_load(collection, vector_field):
    """
    Build an index on the collection and load it into memory.

    Args:
        collection (Collection): The collection to build the index on.
        vector_field (str): The field name of vectors in the collection.

    Returns:
        None
    """
    index_params = {"index_type": "AUTOINDEX", "metric_type": "L2"}
    print("Building AutoIndex...")
    start_time = time.time()
    collection.create_index(field_name=vector_field, index_params=index_params)
    print(f"Index built in {round(time.time() - start_time, 4)} seconds.")

    print("Loading collection into memory...")
    start_time = time.time()
    collection.load()
    print(f"Collection loaded in {round(time.time() - start_time, 4)} seconds.")