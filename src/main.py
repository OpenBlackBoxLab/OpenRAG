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
import time
import json
import os
from pymilvus import DataType, FieldSchema, utility
from openrag.chunk_vectorization import chunk_vectorization
from openrag.text_chunking import text_chunking
from openrag.text_extraction import text_extraction
from openrag.utils import azure_queue_handler, azure_storage_handler
from openrag.vectordb.milvus_adapter import init_milvus_connection
from openrag.vectordb.store_vectors import create_collection_schema, store_vectors
from tqdm import tqdm
import requests

connection_string = "DefaultEndpointsProtocol=https;AccountName=" + os.environ.get("AZURE_STORAGE_ACCOUNT_NAME") + ";AccountKey=" + os.environ.get("AZURE_STORAGE_ACCOUNT_KEY") + ";EndpointSuffix=core.windows.net" # type: ignore
azure_queue_handler = azure_queue_handler.AzureQueueHandler(connection_string, "documents-processing")

processed_documents = []

for message in azure_queue_handler.receive_messages(visibility_timeout=18000):
    try:
        response = requests.get(os.environ.get("ENTITIES_API_URL") + "/parties/" + message.content)

        if response.status_code == 200:
            document = response.json()
        else:
            print("Error: Failed to retrieve the document from the API")
        
        raw_pds_filename = document['data']['file']
        
        start_time = time.time()
        
        print("Processing: " + raw_pds_filename)
        
        text_extraction.extract_and_preprocess_pdf(raw_pds_filename)
        text_chunking.chunk_and_save(raw_pds_filename)
        chunk_vectorization.vectorize_and_store(raw_pds_filename, 'ada', 3072)
        
        azure_queue_handler.delete_message(message)
        processed_documents = processed_documents + [message.content]
        
        print("Processing time: " + str(time.time() - start_time))
        print("=========================================")
    except Exception as e:
        print("Error: " + str(e))
        azure_queue_handler.delete_message(message)
        print("=========================================")

if len(processed_documents) == 0:
    print("No documents to process")
    exit()

try:
    settings = json.loads((azure_storage_handler.download_blob("settings.json", "settings")).decode("utf-8")) # type: ignore

    if (settings['current_collection'] == "vector_collection_politics"):
        collection_name = "vector_collection_politics_tmp"
    else:
        collection_name = "vector_collection_politics"
except Exception as e:
    collection_name = "vector_collection_politics"

vectorized_filenames = azure_storage_handler.list_blobs("vectorized-dicts")
all_vectors = []
all_sources = []
global_indexing = dict()

for vectorized_filename in tqdm(vectorized_filenames, desc="Pushing vectors to Milvus"):
    vectors = azure_storage_handler.get_vectorized_dict(vectorized_filename.split(".")[0])
    
    index_data = dict()
    index_data["len"] = len(vectors)
    index_data["start"] = len(all_vectors)
    
    for vector in vectors:
        all_vectors.append(vector)
        all_sources.append(vectorized_filename.split(".")[0])
        
    index_data["end"] = len(all_vectors)-1
    
    global_indexing[vectorized_filename.split('.')[0]] = index_data

azure_storage_handler.upload_blob("global_indexing.json", "settings", global_indexing)

init_milvus_connection()

if collection_name in utility.list_collections():
    utility.drop_collection(collection_name)
index_field = FieldSchema(name="index", dtype=DataType.INT64, is_primary=True)
vector_field = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=3072)
source_field = FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256)
schema = create_collection_schema([index_field, vector_field, source_field])

store_vectors(collection_name, schema, all_vectors, vector_field.name, all_sources)

settings = dict()
settings["current_collection"] = collection_name
azure_storage_handler.upload_blob("settings.json", "settings", settings)

for document in processed_documents:
    requests.patch(os.environ.get("ENTITIES_API_URL") + "/parties/" + document, json={"state": "available"})