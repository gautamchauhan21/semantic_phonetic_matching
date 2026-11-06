"""CompanyMatch launcher for the semantic_phonetic_matching release (v1.0.0).

Runs the archived scripts unchanged:
    CompanyMatchingWithBERT.py          -> semantic matching
    ColumnMatchingWithBERTPhonetic.py   -> semantic + phonetic matching
Code DOI: https://doi.org/10.5281/zenodo.16221991

Double-click (no arguments): asks for the input file, the Compustat file, and the method.
Command line:
    CompanyMatch <input_file> <compustat_file> [--method semantic|phonetic]

The input and Compustat files are only read. Results are written next to the input file as
<input name>-<YYYYMMDD>-Output.csv and <input name>-<YYYYMMDD>-Log.txt, exactly as the
archived scripts do.
"""
import argparse
import importlib
import multiprocessing
import os
import sys

VERSION = "1.0.0"
DOI = "https://doi.org/10.5281/zenodo.16221991"
METHODS = {
    "semantic": ("CompanyMatchingWithBERT", "company_match"),
    "phonetic": ("ColumnMatchingWithBERTPhonetic", "company_match_phonetic"),
}


def use_bundled_model():
    """Point Hugging Face at the model files shipped inside the program, and stay offline."""
    base = getattr(sys, "_MEIPASS", None)
    if base is None:
        return
    hub = os.path.join(base, "hf_cache", "hub")
    os.environ["HF_HUB_CACHE"] = hub
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


# Runs in the main process and in every worker process before the matching module is imported.
# One OpenMP thread per worker: torch and faiss each ship their own OpenMP library, and with
# several threads per worker the FAISS search on the full Compustat file crashes the workers
# (seen on macOS with torch 2.15.0 and faiss-cpu 1.15.1). The scripts already run one worker
# per CPU core, so this does not slow the matching down.
os.environ.setdefault("OMP_NUM_THREADS", "1")
use_bundled_model()

# Company names can hold characters a Windows console code page cannot show; print a
# placeholder instead of stopping the run. Matching and the output files are unaffected.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass


def ask_with_dialogs():
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.withdraw()
    types = [("Data files", "*.csv *.xls *.xlsx"), ("All files", "*.*")]
    input_file = filedialog.askopenfilename(
        title="Step 1 of 3: select the input file (company names in the first column)", filetypes=types)
    if not input_file:
        return None
    compustat_file = filedialog.askopenfilename(
        title="Step 2 of 3: select the Compustat file (conm, conml, gvkey)", filetypes=types)
    if not compustat_file:
        return None
    phonetic = messagebox.askyesno(
        "Step 3 of 3: method",
        "Add phonetic matching to semantic matching?\n\n"
        "Yes = semantic + phonetic (finds more matches, slightly less precise)\n"
        "No = semantic only (recommended default)")
    root.destroy()
    return input_file, compustat_file, ("phonetic" if phonetic else "semantic")


def run(input_file, compustat_file, method):
    module_name, func_name = METHODS[method]
    print(f"CompanyMatch {VERSION} ({DOI})")
    print(f"Method: {method}\nInput: {input_file}\nCompustat: {compustat_file}\n")
    module = importlib.import_module(module_name)
    try:
        getattr(module, func_name)(input_file, compustat_file)
    except SystemExit as e:  # the archived scripts call sys.exit(1) on errors
        return int(e.code or 0)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Match company names to Compustat gvkeys.")
    parser.add_argument("input_file", nargs="?")
    parser.add_argument("compustat_file", nargs="?")
    parser.add_argument("--method", choices=sorted(METHODS), default="semantic")
    args = parser.parse_args()

    interactive = not (args.input_file and args.compustat_file)
    if interactive:
        picked = ask_with_dialogs()
        if picked is None:
            print("Cancelled.")
            return 1
        input_file, compustat_file, method = picked
    else:
        input_file, compustat_file, method = args.input_file, args.compustat_file, args.method

    code = run(os.path.abspath(input_file), os.path.abspath(compustat_file), method)
    if interactive:
        input("\nPress Enter to close this window.")
    return code


if __name__ == "__main__":
    multiprocessing.freeze_support()  # required so worker processes work inside the .exe
    sys.exit(main())
