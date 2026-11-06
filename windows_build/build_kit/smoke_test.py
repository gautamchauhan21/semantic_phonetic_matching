"""Smoke test for a built CompanyMatch program, using made-up company names only.

Usage: python smoke_test.py <path to CompanyMatch executable>
Creates ./smoke_test_run/ next to this script (made-up data and outputs only), then:
  1. runs the released Python scripts directly (reference),
  2. runs the built program for both methods, offline, with an empty model cache,
  3. requires the program's output to equal the reference, row by row,
  4. requires every exact-name input to receive its made-up gvkey.
Exits 0 only when every check passes. Uses no Compustat or research data.
"""
import glob
import os
import random
import shutil
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
EXE = os.path.abspath(sys.argv[1])
WORK = os.path.join(HERE, "smoke_test_run")
SCRIPTS = {"semantic": "CompanyMatchingWithBERT.py", "phonetic": "ColumnMatchingWithBERTPhonetic.py"}

first = ["Northwind", "Bluecrest", "Ambervale", "Silverpine", "Redfern", "Coralbay", "Ironwood", "Maplegate",
         "Stonebridge", "Clearwater", "Highmoor", "Larkspur", "Brightfield", "Oakhaven", "Westmere", "Foxglove"]
second = ["Lumber", "Biotech", "Logistics", "Foods", "Robotics", "Textiles", "Optics", "Mining",
          "Pharma", "Capital", "Ceramics", "Aviation", "Software", "Plastics", "Marine", "Energy"]
suffix = ["INC", "CORP", "CO", "LTD"]


def make_data():
    rng = random.Random(20261105)
    names = [f"{a} {b}" for a in first for b in second]  # 256 distinct made-up firms
    rows = [{"conm": f"{n.upper()} {rng.choice(suffix)}", "conml": f"{n} {rng.choice(['Inc', 'Corp', 'Co', 'Ltd'])}",
             "gvkey": 900000 + i} for i, n in enumerate(names)]
    comp = pd.DataFrame(rows)
    exact = rng.sample(range(len(names)), 20)
    inputs = [names[i] + rng.choice([" Inc.", " Corporation", " Co", ""]) for i in exact]
    expected = {inp: 900000 + i for inp, i in zip(inputs, exact)}
    typos = [names[i].replace("o", "0", 1).replace("e", "", 1) for i in rng.sample(range(len(names)), 10)]
    noise = ["Zxqv Plorb Holdings", "Quantum Teapot Guild", "Mxyzptlk Sundries", "Vortexian Snail Racing",
             "Glimmerjack Umbrellas", "Hollowquill Paper Kites", "Pebbleworth Sock Exchange",
             "Thistledown Bell Foundry", "Wobbleton Jelly Works", "Cinderlocke Lantern Society"]
    inp = pd.DataFrame({"company name": inputs + typos + noise})
    return comp, inp, expected


def run(cmd, cwd, env, label):
    print(f"--- {label}: {' '.join(os.path.basename(c) for c in cmd[:2])} ...", flush=True)
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, errors="replace", timeout=3600)
    with open(os.path.join(cwd, "console.log"), "w", encoding="utf-8") as f:
        f.write(r.stdout + r.stderr)
    out = glob.glob(os.path.join(cwd, "input-*-Output.csv"))
    print(f"    exit {r.returncode}, output file found: {bool(out)}", flush=True)
    return out[0] if out else None


def main():
    if os.path.isdir(WORK):
        shutil.rmtree(WORK)  # only ever removes this script's own scratch folder
    comp, inp, expected = make_data()
    empty_cache = os.path.join(WORK, "empty_model_cache")
    os.makedirs(empty_cache)
    failures = []
    for method, script in SCRIPTS.items():
        outs = {}
        for kind in ("reference", "program"):
            d = os.path.join(WORK, f"{method}_{kind}")
            os.makedirs(d)
            comp.to_csv(os.path.join(d, "compustat.csv"), index=False)
            inp.to_csv(os.path.join(d, "input.csv"), index=False)
            if kind == "reference":
                env = dict(os.environ, OMP_NUM_THREADS="1")
                cmd = [sys.executable, os.path.join(SRC, script), "input.csv", "compustat.csv"]
            else:
                env = {k: v for k, v in os.environ.items() if k != "OMP_NUM_THREADS"}
                env["HF_HOME"] = empty_cache  # the program must use the model it carries
                cmd = [EXE, "input.csv", "compustat.csv", "--method", method]
            outs[kind] = run(cmd, d, env, f"{method} {kind}")
            if not outs[kind]:
                failures.append(f"{method} {kind}: no output file")
        if all(outs.values()):
            cmp = subprocess.run([sys.executable, os.path.join(HERE, "compare_outputs.py"), outs["reference"], outs["program"]],
                                 capture_output=True, text=True)
            print(cmp.stdout, flush=True)
            if cmp.returncode != 0:
                failures.append(f"{method}: program output differs from reference")
            got = pd.read_csv(outs["program"])
            got = dict(zip(got["company name"], pd.to_numeric(got["gvkey"], errors="coerce")))
            wrong = [n for n, g in expected.items() if got.get(n) != g]
            print(f"    exact-name inputs matched to their made-up gvkey: {len(expected) - len(wrong)} of {len(expected)}")
            if wrong:
                failures.append(f"{method}: {len(wrong)} exact names not matched correctly")
    print("\nSMOKE TEST:", "PASSED" if not failures else "FAILED: " + "; ".join(failures))
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
