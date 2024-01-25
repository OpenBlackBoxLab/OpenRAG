"""
File: azure_helper.py
Author: Nathan Collard <ncollard@openblackbox.be>
Contact: opensource@openblackbox.be
License: MIT License
Project URL: https://github.com/OpenBlackBoxLab/OpenRAG

This file contains helper functions for interacting with Azure Blob Storage. It provides functions for downloading, uploading, and listing blobs in Azure Blob Storage containers. 
Additionally, it includes wrapper functions for specific tasks such as getting raw PDF files, extracting dictionaries, chunking dictionaries, and vectorizing dictionaries.

The functions in this file require the Azure storage connection string to be set as an environment variable named "AZURE_STORAGE_CONNECTION_STRING".

Copyright (c) 2024 Open BlackBox

This file is part of OpenRAG and is released under the MIT License.
See the LICENSE file in the root directory of this project for details.
"""
from azure.storage.blob import BlobServiceClient
from io import BytesIO
import os
import json

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Constants
CONTAINER_NAMES = {
    'raw_pdfs': "raw-pdfs",
    'extracted_dicts': "extracted-dicts",
    'chunked_dicts': "chunked-dicts",
    'vectorized_dicts': "vectorized-dicts"
}
    
def get_blob_service_client():
    """
    Create and return a BlobServiceClient instance using the Azure storage connection string.

    Returns:
        BlobServiceClient: The BlobServiceClient instance.
    """
    connection_string = "DefaultEndpointsProtocol=https;AccountName=" + os.environ.get("AZURE_STORAGE_ACCOUNT_NAME") + ";AccountKey=" + os.environ.get("AZURE_STORAGE_ACCOUNT_KEY") + ";EndpointSuffix=core.windows.net"
    return BlobServiceClient.from_connection_string(connection_string)

def download_blob(file_name, container_name, stream=False):
    """
    Download a blob from the specified Azure Blob Storage container.

    Args:
        file_name (str): The name of the file to be downloaded.
        container_name (str): The name of the Azure Blob Storage container.

    Returns:
        BytesIO: A BytesIO stream of the downloaded blob.
    """
    blob_client = get_blob_service_client().get_blob_client(container=container_name, blob=file_name)
    blob_data = blob_client.download_blob().readall()
    return BytesIO(blob_data) if stream else blob_data

def list_blobs(container_name):
    """
    List the names of all blobs in the specified Azure Blob Storage container.

    Args:
        container_name (str): The name of the Azure Blob Storage container.

    Returns:
        list: A list of blob names.
    """
    container_client = get_blob_service_client().get_container_client(container_name)
    return [blob.name for blob in container_client.list_blobs()]

def upload_blob(file_name, container_name, data):
    """
    Upload data to a blob in the specified Azure Blob Storage container.

    Args:
        file_name (str): The name of the file to be uploaded.
        container_name (str): The name of the Azure Blob Storage container.
        data (dict): The data to be uploaded.

    Returns:
        bool: True if upload is successful, False otherwise.
    """
    blob_client = get_blob_service_client().get_blob_client(container=container_name, blob=file_name)
    try:
        blob_client.upload_blob(json.dumps(data), overwrite=True)
        return True
    except Exception as e:
        print(f"Error while uploading to Azure Blob Storage: {e}")
        return False

# Wrapper functions for specific tasks
def get_raw_pdf(file_name):
    """
    Get a raw PDF file from Azure Blob Storage.

    Args:
        file_name (str): The name of the file to extract the text from.

    Returns:
        BytesIO: The stream of the PDF file.
    """
    return download_blob(f"{file_name}.pdf", CONTAINER_NAMES['raw_pdfs'], stream=True)

def get_extracted_dict(file_name):
    """
    Get the extracted dictionary from a JSON file in Azure Blob Storage.

    Args:
        file_name (str): The name of the file.

    Returns:
        dict: The content of the JSON file.
    """
    blob_stream = download_blob(f"{file_name}.json", CONTAINER_NAMES['extracted_dicts'])
    return json.loads(blob_stream.decode("utf-8"))

def put_extracted_dict(file_name, data):
    """
    Upload the extracted dictionary to a JSON file in Azure Blob Storage.

    Args:
        file_name (str): The name of the original file.
        data (dict): The extracted dictionary.

    Returns:
        bool: True if the upload is successful, False otherwise.
    """
    return upload_blob(f"{file_name}.json", CONTAINER_NAMES['extracted_dicts'], data)

def get_chunked_dict(file_name):
    """
    Get the chunked dictionary from a JSON file in Azure Blob Storage.

    Args:
        file_name (str): The name of the file.

    Returns:
        dict: The content of the JSON file.
    """
    blob_stream = download_blob(f"{file_name}.json", CONTAINER_NAMES['chunked_dicts'])
    return json.loads(blob_stream.decode("utf-8"))

def put_chunked_dict(file_name, chunks_dict):
    """
    Upload the chunked dictionary to a JSON file in Azure Blob Storage.

    Args:
        file_name (str): The name of the original file.
        chunks_dict (dict): The chunked dictionary.

    Returns:
        bool: True if the upload is successful, False otherwise.
    """
    return upload_blob(f"{file_name}.json", CONTAINER_NAMES['chunked_dicts'], chunks_dict)

def get_vectorized_dict(file_name):
    """
    Get the vectorized dictionary from a JSON file in Azure Blob Storage.

    Args:
        file_name (str): The name of the file.

    Returns:
        dict: The content of the JSON file.
    """
    blob_stream = download_blob(f"{file_name}.json", CONTAINER_NAMES['vectorized_dicts'])
    return json.loads(blob_stream.decode("utf-8"))

def put_vectorized_dict(file_name, vectorized_dict):
    """
    Upload the vectorized dictionary to a JSON file in Azure Blob Storage.

    Args:
        file_name (str): The name of the original file.
        vectorized_dict (dict): The vectorized dictionary.

    Returns:
        bool: True if the upload is successful, False otherwise.
    """
    return upload_blob(f"{file_name}.json", CONTAINER_NAMES['vectorized_dicts'], vectorized_dict)
