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
    """Read the first column of input data from XLS, XLSX, or CSV files and rename it to 'company_name'."""
    try:
        if file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path, usecols=[0])
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path, usecols=[0])
        else:
            raise ValueError("Unsupported file format. Please provide a .xls, .xlsx, or .csv file.")
        # Rename the first column to 'company_name' for consistent processing
        df.columns = ['company_name']
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {file_path}")

def read_compustat_data(file_path):
    """Read Compustat data from xls, xlsx, or csv files."""
    try:
        if file_path.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a .xls, .xlsx, or .csv file.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Compustat file not found: {file_path}")

def clean_company_name(name):
    """Clean company name by normalizing case, removing punctuation, and trimming suffixes."""
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    suffixes = ['inc', 'corp', 'corporation', 'co', 'company', 'ltd', 'limited']
    words = name.split()
    if words and words[-1] in suffixes:
        name = ' '.join(words[:-1])
    return name

# Initialize SentenceTransformer model
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = model.to(device)
except Exception as e:
    raise RuntimeError(f"Failed to initialize SentenceTransformer model: {e}")

def get_bert_embeddings(texts, batch_size=32):
    """Generate BERT embeddings for a list of texts in batches."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch, device=device, batch_size=batch_size)
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

def process_chunk(chunk_data, compustat_data, compustat_embeddings_conm, compustat_embeddings_conml, compustat_clean_names_conm, compustat_clean_names_conml, score_cutoff=0.90):
    """Process a chunk of company names to find matches using FAISS."""
    results = []

    # Build FAISS indices for conm and conml embeddings
    d = compustat_embeddings_conm.shape[1]
    index_conm = faiss.IndexFlatL2(d)
    index_conml = faiss.IndexFlatL2(d)
    index_conm.add(compustat_embeddings_conm.astype(np.float32))
    index_conml.add(compustat_embeddings_conml.astype(np.float32))

    # Clean and encode chunk of company names
    clean_inputs = [clean_company_name(name) for name in chunk_data]
    input_embeddings = get_bert_embeddings(clean_inputs)

    for i, (company_name, clean_input, input_embedding) in enumerate(zip(chunk_data, clean_inputs, input_embeddings)):
        # Search FAISS indices for nearest neighbors
        D_conm, I_conm = index_conm.search(np.array([input_embedding]).astype(np.float32), 1)
        D_conml, I_conml = index_conml.search(np.array([input_embedding]).astype(np.float32), 1)

        # Convert L2 distance to cosine similarity
        sim_conm = 1 - D_conm[0][0] / 2
        sim_conml = 1 - D_conml[0][0] / 2

        # Boost similarity for exact substring matches
        if clean_input and compustat_clean_names_conm[I_conm[0][0]] and clean_input in compustat_clean_names_conm[I_conm[0][0]]:
            sim_conm = min(sim_conm + 0.1, 1.0)
        if clean_input and compustat_clean_names_conml[I_conml[0][0]] and clean_input in compustat_clean_names_conml[I_conml[0][0]]:
            sim_conml = min(sim_conml + 0.1, 1.0)

        # Select best match based on similarity
        max_sim = max(sim_conm, sim_conml)
        max_index = I_conm[0][0] if sim_conm >= sim_conml else I_conml[0][0]

        if max_sim >= score_cutoff:
            best_match = compustat_data['conm'].iloc[max_index]
            confidence = round(max_sim * 100, 2)
            matched_conml = compustat_data.at[max_index, 'conml']
            matched_gvkey = compustat_data.at[max_index, 'gvkey']
            results.append([company_name, best_match, matched_conml, matched_gvkey, confidence])
            print(f"Matched '{company_name}' with '{best_match}' (Confidence: {confidence:.2f}%)")
        else:
            results.append([company_name, None, None, None, None])
            print(f"No match found for '{company_name}'")

    return results

def company_match(input_file_path, compustat_file_path):
    """Match company names from input file with Compustat data using parallel processing."""
    try:
        # Read input and Compustat data
        input_data = read_input_data(input_file_path)
        compustat_data = read_compustat_data(compustat_file_path)

        # Deduplicate Compustat data by gvkey and conm
        compustat_data = compustat_data.drop_duplicates(subset=['gvkey', 'conm']).reset_index(drop=True)

        column_name = input_data.columns[0]
        company_names = input_data[column_name].tolist()

        # Clean Compustat company names
        print("Cleaning Compustat names...")
        compustat_clean_names_conm = [clean_company_name(name) for name in compustat_data['conm']]
        compustat_clean_names_conml = [clean_company_name(name) for name in compustat_data['conml']]

        # Precompute BERT embeddings for Compustat data
        print("Precomputing BERT embeddings for Compustat data...")
        compustat_embeddings_conm = get_bert_embeddings(compustat_clean_names_conm)
        compustat_embeddings_conml = get_bert_embeddings(compustat_clean_names_conml)
        print("Embeddings computed.")

        # Set up log file
        today = datetime.now().strftime("%Y%m%d")
        log_file = f"{os.path.splitext(input_file_path)[0]}-{today}-Log.txt"
        with open(log_file, "w") as log:
            log.write(f"Total rows to process: {len(company_names)}\n")
            log.write(f"Compustat unique rows: {len(compustat_data)}\n")

        # Split company names into chunks for parallel processing
        num_cores = cpu_count()
        chunk_size = max(1, len(company_names) // num_cores)
        chunks = [company_names[i:i + chunk_size] for i in range(0, len(company_names), chunk_size)]

        # Process chunks in parallel
        print(f"Processing {len(company_names)} names using {num_cores} cores...")
        partial_process = partial(
            process_chunk,
            compustat_data=compustat_data,
            compustat_embeddings_conm=compustat_embeddings_conm,
            compustat_embeddings_conml=compustat_embeddings_conml,
            compustat_clean_names_conm=compustat_clean_names_conm,
            compustat_clean_names_conml=compustat_clean_names_conml
        )

        with Pool(num_cores) as pool:
            results = pool.map(partial_process, chunks)

        # Flatten results
        results = [item for sublist in results for item in sublist]

        # Log completion
        with open(log_file, "a") as log:
            log.write(f"Completed processing of {len(company_names)} rows.\n")

        # Save results to CSV
        output_file = f"{os.path.splitext(input_file_path)[0]}-{today}-Output.csv"
        output_df = pd.DataFrame(results, columns=["company name", "conm", "conml", "gvkey", "confidence"])
        output_df.to_csv(output_file, index=False)

        print(f"\nProcessing completed. Output saved to {output_file} and log saved to {log_file}.")
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)

def main():
    """Parse command-line arguments and run the company matching process."""
    parser = argparse.ArgumentParser(description="Match company names to Compustat data using BERT and FAISS.")
    parser.add_argument("input_file", help="Path to input file (csv, xls, or xlsx)")
    parser.add_argument("compustat_file", help="Path to Compustat file (csv, xls, or xlsx)")
    args = parser.parse_args()

    company_match(args.input_file, args.compustat_file)

if __name__ == '__main__':
    main()
