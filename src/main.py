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
from openrag.utils import azure_helper
from openrag.text_extraction import text_extraction
from openrag.text_chunking import text_chunking
from openrag.chunk_vectorization import chunk_vectorization

raw_pds_filenames = azure_helper.list_blobs("raw-pdfs")

for raw_pds_filename in raw_pds_filenames:
    raw_pds_filename = raw_pds_filename.split(".")[0]
    print(raw_pds_filename)
    
    # text_extraction.extract_and_preprocess_pdf(raw_pds_filename)
    # text_chunking.chunk_and_save(raw_pds_filename)
    # chunk_vectorization.vectorize_and_store(raw_pds_filename, 'ada', 1536)
    
    