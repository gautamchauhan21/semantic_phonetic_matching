"""Company Name Matching Script

This script matches company names from an input file to Compustat data using BERT embeddings
and FAISS for efficient similarity search. It supports parallel processing and logs results.

Usage:
    python company_match.py <input_file> <compustat_file>

Dependencies:
    See requirements.txt for required packages.
"""
import pandas as pd
from datetime import datetime
import os
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import faiss
from multiprocessing import Pool, cpu_count
from functools import partial
import argparse
import sys

def read_input_data(file_path):
    """
    Read the first column of input data from XLS, XLSX, or CSV files and rename it to 'company_name'.

    Args:
        file_path (str): The path to the input file.

    Returns:
        pd.DataFrame: A DataFrame containing the first column renamed to 'company_name'.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file format is not supported (.xls, .xlsx, or .csv).
    """
    try:
        # Check file extension to determine the appropriate pandas read function
        if file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path, usecols=[0]) # Read only the first column
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path, usecols=[0]) # Read only the first column
        else:
            raise ValueError("Unsupported file format. Please provide a .xls, .xlsx, or .csv file.")

        # Rename the first column to 'company_name' for consistent processing across different input files
        df.columns = ['company_name']
        return df
    except FileNotFoundError:
        # Re-raise with a more specific error message for clarity
        raise FileNotFoundError(f"Input file not found: {file_path}")

def read_compustat_data(file_path):
    """
    Read Compustat data from XLS, XLSX, or CSV files.

    Args:
        file_path (str): The path to the Compustat data file.

    Returns:
        pd.DataFrame: A DataFrame containing the Compustat data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file format is not supported (.xls, .xlsx, or .csv).
    """
    try:
        # Check file extension to determine the appropriate pandas read function
        if file_path.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a .xls, .xlsx, or .csv file.")
    except FileNotFoundError:
        # Re-raise with a more specific error message for clarity
        raise FileNotFoundError(f"Compustat file not found: {file_path}")

def clean_company_name(name):
    """
    Clean company name by converting to lowercase, removing punctuation,
    normalizing whitespace, and trimming common company suffixes.

    Args:
        name (str): The company name to clean.

    Returns:
        str: The cleaned company name. Returns an empty string if input is not a string.
    """
    if not isinstance(name, str):
        return "" # Handle non-string inputs gracefully

    name = name.lower() # Convert to lowercase for case-insensitive matching
    name = re.sub(r'[^\w\s]', '', name) # Remove all punctuation (non-alphanumeric, non-whitespace characters)
    name = re.sub(r'\s+', ' ', name).strip() # Replace multiple spaces with single space and strip leading/trailing whitespace

    # List of common suffixes to remove for better normalization
    suffixes = ['inc', 'corp', 'corporation', 'co', 'company', 'ltd', 'limited']
    words = name.split()

    # If the last word is a common suffix, remove it
    if words and words[-1] in suffixes:
        name = ' '.join(words[:-1])
    return name

def normalize_embeddings(embeddings):
    """
    Normalize embeddings to unit length. This is crucial for cosine similarity
    calculations, as L2 distance between normalized vectors directly relates to
    cosine similarity (D = 2 - 2*CosineSimilarity).

    Args:
        embeddings (np.ndarray): A 2D numpy array of embeddings.

    Returns:
        np.ndarray: Normalized embeddings. Handles zero-norm vectors by preventing division by zero.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Divide embeddings by their L2 norm. If a norm is zero (all zeros embedding),
    # replace 0 with 1 in the divisor to avoid division by zero, resulting in zero embedding.
    return embeddings / np.where(norms == 0, 1, norms)

# Initialize SentenceTransformer model globally for efficiency
# This model will be loaded once and used by all processes (though copied for each process)
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Determine the device to run the model on (MPS for Apple Silicon, otherwise CPU)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = model.to(device) # Move the model to the selected device
    print(f"SentenceTransformer model loaded and moved to {device}.")
except Exception as e:
    # If model initialization fails, terminate the program with an error
    raise RuntimeError(f"Failed to initialize SentenceTransformer model: {e}")

def get_bert_embeddings(texts, batch_size=32):
    """
    Generate BERT embeddings for a list of texts in batches.

    Args:
        texts (list): A list of strings for which to generate embeddings.
        batch_size (int): The number of texts to process in each batch.

    Returns:
        np.ndarray: A 2D numpy array where each row is the embedding for a text.
    """
    embeddings = []
    # Process texts in batches to optimize GPU/CPU usage and memory
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Encode the batch using the pre-loaded SentenceTransformer model
        batch_embeddings = model.encode(batch, device=device, batch_size=batch_size, show_progress_bar=False)
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings) # Stack all batch embeddings into a single numpy array

def process_chunk(chunk_data, compustat_data, compustat_embeddings_conm, compustat_embeddings_conml, compustat_clean_names_conm, compustat_clean_names_conml):
    """
    Process a chunk of company names using pre-built FAISS indices for similarity search.
    This function is designed to be run in parallel across multiple CPU cores.

    Args:
        chunk_data (list): A list of company names from the input file to process in this chunk.
        compustat_data (pd.DataFrame): The full Compustat DataFrame.
        compustat_embeddings_conm (np.ndarray): Normalized BERT embeddings for 'conm' names.
        compustat_embeddings_conml (np.ndarray): Normalized BERT embeddings for 'conml' names.
        compustat_clean_names_conm (list): Cleaned 'conm' names from Compustat.
        compustat_clean_names_conml (list): Cleaned 'conml' names from Compustat.

    Returns:
        list: A list of lists, where each inner list contains match results for one company.
    """
    results = []

    # Initialize FAISS indices within each worker process.
    # While this means re-initializing for each chunk, the embeddings are passed in,
    # and for multiprocessing.Pool, this is often simpler than managing shared FAISS indices.
    d = compustat_embeddings_conm.shape[1] # Get embedding dimension
    index_conm = faiss.IndexFlatL2(d) # Create L2 distance index for 'conm'
    index_conml = faiss.IndexFlatL2(d) # Create L2 distance index for 'conml'

    # Add all Compustat embeddings to their respective FAISS indices
    # FAISS requires float32 type
    index_conm.add(compustat_embeddings_conm.astype(np.float32))
    index_conml.add(compustat_embeddings_conml.astype(np.float32))

    # Clean and generate BERT embeddings for the current chunk of input company names
    clean_inputs = [clean_company_name(name) for name in chunk_data]
    input_embeddings = get_bert_embeddings(clean_inputs)
    # Normalize input embeddings to unit length for accurate cosine similarity with FAISS L2 search
    input_embeddings = normalize_embeddings(input_embeddings)

    # Process each company name in the current chunk
    for i, (company_name, clean_input, input_embedding) in enumerate(zip(chunk_data, clean_inputs, input_embeddings)):
        # Search for the nearest neighbor in both 'conm' and 'conml' FAISS indices
        # input_embedding must be reshaped to (1, embedding_dim) for FAISS search
        D_conm, I_conm = index_conm.search(np.array([input_embedding]).astype(np.float32), 1) # Search for 1 nearest neighbor
        D_conml, I_conml = index_conml.search(np.array([input_embedding]).astype(np.float32), 1)

        # Convert L2 distance (D) to cosine similarity (S). For normalized embeddings, S = 1 - D/2.
        sim_conm = 1 - D_conm[0][0] / 2
        sim_conml = 1 - D_conml[0][0] / 2

        # Apply a similarity boost if there's an exact substring match in the cleaned names.
        # This gives higher confidence to strong textual matches.
        if clean_input and compustat_clean_names_conm[I_conm[0][0]] and clean_input in compustat_clean_names_conm[I_conm[0][0]]:
            sim_conm = min(sim_conm + 0.1, 1.0) # Boost by 0.1, but cap at 1.0
        if clean_input and compustat_clean_names_conml[I_conml[0][0]] and clean_input in compustat_clean_names_conml[I_conml[0][0]]:
            sim_conml = min(sim_conml + 0.1, 1.0) # Boost by 0.1, but cap at 1.0

        # Select the best match (highest similarity) between 'conm' and 'conml'
        max_sim = max(sim_conm, sim_conml)
        max_index = I_conm[0][0] if sim_conm >= sim_conml else I_conml[0][0] # Get the index of the best match

        # If the highest similarity meets the cutoff, record the match details
        if max_sim >= score_cutoff:
            best_match = compustat_data['conm'].iloc[max_index] # Get the original 'conm' name
            confidence = round(max_sim * 100, 2) # Convert similarity to percentage and round to 2 decimal places
            matched_conml = compustat_data.at[max_index, 'conml'] # Get original 'conml'
            matched_gvkey = compustat_data.at[max_index, 'gvkey'] # Get original 'gvkey'
            results.append([company_name, best_match, matched_conml, matched_gvkey, confidence])
            # Print success message (note: this will interleave from parallel processes)
            print(f"Matched '{company_name}' with '{best_match}' (Confidence: {confidence:.2f}%)")
        else:
            # If no match meets the cutoff, record None for match details
            results.append([company_name, None, None, None, None])
            print(f"No match found for '{company_name}'") # Print no match message

    return results

def company_match(input_file_path, compustat_file_path):
    """
    Match company names from an input file with Compustat data using BERT embeddings
    and FAISS for efficient similarity search. Leverages parallel processing for speed.

    Args:
        input_file_path (str): Path to the file containing company names to match.
        compustat_file_path (str): Path to the Compustat database file.
    """
    try:
        # Load input company names and Compustat financial data
        input_data = read_input_data(input_file_path)
        compustat_data = read_compustat_data(compustat_file_path)

        # Deduplicate Compustat data based on 'gvkey' and 'conm' to ensure unique entities
        compustat_data = compustat_data.drop_duplicates(subset=['gvkey', 'conm']).reset_index(drop=True)

        # Get the list of company names to be matched from the input DataFrame
        company_names = input_data['company_name'].tolist() # Use the standardized 'company_name' column

        # Clean Compustat company names ('conm' and 'conml' columns)
        print("Cleaning Compustat names...")
        compustat_clean_names_conm = [clean_company_name(name) for name in compustat_data['conm']]
        compustat_clean_names_conml = [clean_company_name(name) for name in compustat_data['conml']]

        # Precompute BERT embeddings for all cleaned Compustat names
        # These embeddings will be used for FAISS indexing
        print("Precomputing BERT embeddings for Compustat data...")
        raw_compustat_embeddings_conm = get_bert_embeddings(compustat_clean_names_conm)
        raw_compustat_embeddings_conml = get_bert_embeddings(compustat_clean_names_conml)

        # Apply normalization to Compustat embeddings. This is critical for L2 distance
        # in FAISS to correctly represent cosine similarity.
        compustat_embeddings_conm = normalize_embeddings(raw_compustat_embeddings_conm)
        compustat_embeddings_conml = normalize_embeddings(raw_compustat_embeddings_conml)
        print("Embeddings computed and normalized.")

        # Prepare log file path with current date for tracking
        today = datetime.now().strftime("%Y%m%d")
        log_file = f"{os.path.splitext(input_file_path)[0]}-{today}-Log.txt"
        with open(log_file, "w") as log:
            log.write(f"Total rows to process: {len(company_names)}\n")
            log.write(f"Compustat unique rows: {len(compustat_data)}\n")
            log.write(f"Similarity Score Cutoff: {score_cutoff}\n")


        # Determine the number of CPU cores available for parallel processing
        num_cores = cpu_count()
        # Calculate chunk size for distributing the workload among processes
        # Ensure chunk_size is at least 1 to avoid empty chunks
        chunk_size = max(1, len(company_names) // num_cores)
        # Divide the list of input company names into smaller chunks
        chunks = [company_names[i:i + chunk_size] for i in range(0, len(company_names), chunk_size)]

        # Prepare a partial function for 'process_chunk' to pass common arguments
        # This avoids passing large objects multiple times for each call in the pool map
        print(f"Processing {len(company_names)} names using {num_cores} cores...")
        partial_process = partial(
            process_chunk,
            compustat_data=compustat_data, # Full DataFrame for retrieving original company names
            compustat_embeddings_conm=compustat_embeddings_conm,
            compustat_embeddings_conml=compustat_embeddings_conml,
            compustat_clean_names_conm=compustat_clean_names_conm,
            compustat_clean_names_conml=compustat_clean_names_conml,
            score_cutoff=0.85 # Pass the configurable cutoff
        )

        # Use a multiprocessing Pool to execute 'process_chunk' in parallel
        with Pool(num_cores) as pool:
            # map applies partial_process to each chunk in 'chunks'
            results = pool.map(partial_process, chunks)

        # Flatten the list of lists (results from each chunk) into a single list of results
        results = [item for sublist in results for item in sublist]

        # Log completion time and total processed rows
        with open(log_file, "a") as log:
            log.write(f"Completed processing of {len(company_names)} rows.\n")
            log.write(f"Processing finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


        # Define the output CSV file path with current date
        output_file = f"{os.path.splitext(input_file_path)[0]}-{today}-Output.csv"
        # Create a pandas DataFrame from the collected results
        output_df = pd.DataFrame(results, columns=["company name", "conm", "conml", "gvkey", "Confidence"])
        # Save the results DataFrame to a CSV file
        output_df.to_csv(output_file, index=False)

        print(f"\nProcessing completed. Output saved to {output_file} and log saved to {log_file}.")
    except Exception as e:
        # Catch any exceptions during the main process and print an error message
        print(f"Error during processing: {e}")
        sys.exit(1) # Exit the script with an error code


def main():
    """Parse command-line arguments and run the company matching process."""
    parser = argparse.ArgumentParser(description="Match company names to Compustat data using BERT and FAISS.")
    parser.add_argument("input_file", help="Path to input file (csv, xls, or xlsx)")
    parser.add_argument("compustat_file", help="Path to Compustat file (csv, xls, or xlsx)")
    args = parser.parse_args()

    company_match(args.input_file, args.compustat_file)

if __name__ == '__main__':
    main()
