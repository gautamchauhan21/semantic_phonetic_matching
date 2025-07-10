# Company Name Matching

This repository contains two Python scripts for matching company names from an input file to Compustat data using BERT embeddings and FAISS for efficient similarity search. One script includes phonetic matching with Soundex for improved accuracy.

## Features

- Matches company names using BERT-based embeddings (`all-MiniLM-L6-v2`).
- Utilizes FAISS for fast similarity search.
- Supports parallel processing for efficiency.
- Handles input files in CSV, XLS, or XLSX formats.
- `company_match_phonetic.py` incorporates Soundex for phonetic similarity.
- Generates output CSV with matched results and confidence scores.
- Logs processing details to a timestamped log file.

## Prerequisites

- Python 3.8+
- Required packages listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/company-name-matching.git
   cd company-name-matching
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run either script with input and Compustat file paths as arguments:

```bash
python company_match.py <input_file> <compustat_file>
```

or

```bash
python company_match_phonetic.py <input_file> <compustat_file>
```

- `<input_file>`: Path to CSV, XLS, or XLSX file with company names (first column).
- `<compustat_file>`: Path to Compustat data file (CSV, XLS, or XLSX).

Example:
```bash
python company_match.py data/CUS_NAME_MATCH.csv data/Compustat.xlsx
```

## Output

- **Output CSV**: `<input_file_name>-YYYYMMDD-Output.csv` with columns:
  - Input company name
  - Matched Compustat name (`conm`)
  - Compustat long name (`conml`)
  - Compustat `gvkey`
  - Confidence score (%)
- **Log File**: `<input_file_name>-YYYYMMDD-Log.txt` with processing details.

## Scripts

- `company_match.py`: Matches company names using BERT embeddings and FAISS, with substring matching boosts.
- `company_match_phonetic.py`: Extends `company_match.py` with Soundex-based phonetic matching for improved accuracy.

## Requirements

- pandas>=2.0.0
- torch>=2.0.0
- sentence-transformers>=2.2.0
- numpy>=1.24.0
- faiss-cpu>=1.7.0
- jellyfish>=0.8.0 (for `company_match_phonetic.py`)
- openpyxl>=3.0.0

## Notes

- Ensure input files have company names in the first column.
- Compustat data must include `gvkey`, `conm`, and `conml` columns.
- The scripts handle deduplication of Compustat data by `gvkey` and `conm`.
- For large datasets, parallel processing leverages all available CPU cores.

## License

Apache 2.0 License. See LICENSE file for details.
