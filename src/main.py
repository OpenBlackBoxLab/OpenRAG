"""
File: main.py
Author: Nathan Collard <ncollard@openblackbox.be>
Contact: opensource@openblackbox.be
License: MIT License
Project URL: https://github.com/OpenBlackBoxLab/OpenRAG

Brief description of this file's purpose.

Copyright (c) 2024 Open BlackBox

This file is part of OpenRAG and is released under the MIT License.
See the LICENSE file in the root directory of this project for details.
"""
from pymilvus import DataType, FieldSchema, utility
from openrag.chunk_vectorization import chunk_vectorization
from openrag.text_chunking import text_chunking
from openrag.text_extraction import text_extraction
from openrag.utils import azure_helper
from openrag.vectordb.milvus_adapter import init_milvus_connection
from openrag.vectordb.store_vectors import create_collection_schema, store_vectors

raw_pds_filenames = azure_helper.list_blobs("raw-pdfs")

for raw_pds_filename in raw_pds_filenames:
    raw_pds_filename = raw_pds_filename.split(".")[0]
    print(raw_pds_filename)
    
    text_extraction.extract_and_preprocess_pdf(raw_pds_filename)
    text_chunking.chunk_and_save(raw_pds_filename)
    chunk_vectorization.vectorize_and_store(raw_pds_filename, 'ada', 3072)

vectorized_filenames = azure_helper.list_blobs("vectorized-dicts")
all_vectors = []
all_sources = []
global_indexing = dict()

for vectorized_filename in vectorized_filenames:
    vectors = azure_helper.get_vectorized_dict(vectorized_filename.split(".")[0])
    
    index_data = dict()
    index_data["len"] = len(vectors)
    index_data["start"] = len(all_vectors)
    
    for vector in vectors:
        all_vectors.append(vector)
        all_sources.append(vectorized_filename.split(".")[0])
        
    index_data["end"] = len(all_vectors)-1
    
    global_indexing[vectorized_filename] = index_data

print(global_indexing)

init_milvus_connection()

if "vector_collection_politics" in utility.list_collections():
    utility.drop_collection("vector_collection_politics")
index_field = FieldSchema(name="index", dtype=DataType.INT64, is_primary=True)
vector_field = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=3072)
source_field = FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256)
schema = create_collection_schema([index_field, vector_field, source_field])

store_vectors("vector_collection_politics", schema, all_vectors, vector_field.name, all_sources)