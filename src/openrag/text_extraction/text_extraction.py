"""
File: text_extraction.py
Author: Nathan Collard <ncollard@openblackbox.be>
Contact: opensource@openblackbox.be
License: MIT License
Project URL: https://github.com/OpenBlackBoxLab/OpenRAG

This file contains functions for extracting text from a PDF file, preprocessing the extracted text, and saving the processed text to a JSON file. It uses the PyMuPDF library for PDF text extraction.

Functions:
- extract_pdf_text(file_name): Extracts text from a PDF file and returns a list of strings, where each string is the text of a page.
- preprocess_text(text): Preprocesses the extracted text by replacing special characters, combining lines within the same paragraph, and removing lines that do not contain any letters.
- replace_special_characters(text): Replaces special characters and extra spaces in the text.
- combine_lines(text): Combines lines within the same paragraph.
- remove_non_letter_lines(text): Removes lines that do not contain any letters.
- save_text_to_json(pages_text, file_name): Saves the processed text to a JSON file via Azure handler.
- extract_and_process_pdf(file_name): Extracts text from a PDF file, preprocesses it, and saves it to a JSON file.

Copyright (c) 2024 Open BlackBox

This file is part of OpenRAG and is released under the MIT License.
See the LICENSE file in the root directory of this project for details.
"""
import fitz
import re
from tqdm import tqdm
from ..utils import azure_helper as azure_helper

def extract_pdf_text(file_name):
    """
    Extract text from a PDF file using PyMuPDF.

    Args:
        file_name (str): The name of the PDF file to extract text from.

    Returns:
        list: A list of strings, where each string is the text of a page.
    """
    pdf_document = fitz.open("pdf", azure_helper.get_raw_pdf(file_name)) # type: ignore
    pages_text = [pdf_document.load_page(page_number).get_text() for page_number in tqdm(range(len(pdf_document)), desc="Extracting")]
    pdf_document.close()
    return pages_text

def preprocess_text(text):
    """
    Preprocess the extracted text.

    Args:
        text (str): The text to be preprocessed.

    Returns:
        str: The preprocessed text.
    """
    text = replace_special_characters(text)
    text = combine_lines(text)
    text = remove_non_letter_lines(text)
    return text

def replace_special_characters(text):
    """
    Replace special characters and extra spaces in the text.

    Args:
        text (str): The text with potential special characters.

    Returns:
        str: Text with special characters replaced.
    """
    replacements = {
        '\xa0': ' ',  # Replace non-breaking spaces
        '- ': '',     # Handle hyphenated line breaks
        '•': '',      # Remove bullet points
        '●': '',      # Remove bullet points
        ' +': ' '     # Replace multiple spaces with a single space
    }
    for old, new in replacements.items():
        text = re.sub(old, new, text)
    return text

def combine_lines(text):
    """
    Combine lines within the same paragraph.

    Args:
        text (str): The text to be combined.

    Returns:
        str: Combined text content.
    """
    return " ".join(text.splitlines())

def remove_non_letter_lines(text):
    """
    Remove lines that do not contain any letters.

    Args:
        text (str): The text with lines to be removed.

    Returns:
        str: Text with non-letter lines removed.
    """
    return "\n".join([line for line in text.splitlines() if re.search(r'[a-zA-Z]', line)])

def save_text_to_json(pages_text, file_name):
    """
    Save the processed text to a JSON file via Azure handler.

    Args:
        pages_text (list): A list of processed text strings, one per page.
        file_name (str): The name of the file to save the JSON to.
    """
    data = [{"text": text, "page": index + 1} for index, text in enumerate(pages_text)]
    azure_helper.put_extracted_dict(file_name, data)

def extract_and_preprocess_pdf(file_name):
    """
    Extract text from a PDF file, preprocess it, and save to JSON.

    Args:
        file_name (str): The name of the PDF file to process.
    """
    pages_text = extract_pdf_text(file_name)
    processed_text = [preprocess_text(page) for page in pages_text]
    save_text_to_json(processed_text, file_name)