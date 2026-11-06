"""Compare two CompanyMatch output CSVs row by row.

Usage: python compare_outputs.py <reference_output.csv> <test_output.csv>
Reads both files only. Prints a short report and exits 0 when every row has the same
company name, conm, conml and gvkey and the confidence differs by at most 0.01.
"""
import sys

import pandas as pd

ref = pd.read_csv(sys.argv[1], dtype=str, keep_default_na=False)
new = pd.read_csv(sys.argv[2], dtype=str, keep_default_na=False)
print(f"reference: {sys.argv[1]}\ntest:      {sys.argv[2]}")
print(f"rows: {len(ref)} vs {len(new)} | columns: {list(ref.columns)} vs {list(new.columns)}")
ok = len(ref) == len(new) and list(ref.columns) == list(new.columns)
if ok:
    for col in ["company name", "conm", "conml", "gvkey"]:
        diff = (ref[col] != new[col]).sum()
        print(f"{col:13} rows that differ: {diff}")
        ok &= diff == 0
    a = pd.to_numeric(ref["confidence"], errors="coerce")
    b = pd.to_numeric(new["confidence"], errors="coerce")
    both_blank = a.isna() & b.isna()
    gap = (a - b).abs().where(~both_blank, 0)
    blank_mismatch = (a.isna() != b.isna()).sum()
    print(f"confidence    max difference: {gap.max():.4f} | matched-vs-unmatched mismatches: {blank_mismatch}")
    print(f"matched rows: {a.notna().sum()} vs {b.notna().sum()}")
    ok &= blank_mismatch == 0 and gap.max() <= 0.01
    print(f"byte-identical files: {open(sys.argv[1], 'rb').read() == open(sys.argv[2], 'rb').read()}")
print("RESULT:", "SAME" if ok else "DIFFERENT")
sys.exit(0 if ok else 1)
