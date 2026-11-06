# CompanyMatch for Windows

A double-click Windows program that runs the v1.0.0 matching scripts unchanged
(`CompanyMatchingWithBERT.py` and `ColumnMatchingWithBERTPhonetic.py`, archived at
https://doi.org/10.5281/zenodo.16221991). The all-MiniLM-L6-v2 model is included, so it works offline.

## Use

1. Unzip `CompanyMatch-v1.0.0-windows.zip` and open the `CompanyMatch` folder.
2. Double-click `CompanyMatch.exe`.
3. Pick the input file (company names in the first column; CSV, XLS or XLSX), then the Compustat file
   (`conm`, `conml`, `gvkey`), then the method (semantic, or semantic + phonetic).
4. Results appear next to the input file: `<input name>-<YYYYMMDD>-Output.csv` and `-Log.txt`.

Command line: `CompanyMatch.exe <input_file> <compustat_file> --method semantic|phonetic`

## Build

The `Build Windows EXE` GitHub Actions workflow builds it from this branch. To build by hand on 64-bit
Windows with Python 3.13: copy the two v1.0.0 scripts into `src/`, then run `build_kit\build_windows.bat`.
`build_kit\smoke_test.py` checks the program against the scripts using made-up company names.
