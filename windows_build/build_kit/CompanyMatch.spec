# PyInstaller recipe for CompanyMatch (one-folder build: CompanyMatch/CompanyMatch.exe plus its files).
# Build with:  pyinstaller --noconfirm CompanyMatch.spec   (after running prepare_model.py)
import os
from PyInstaller.utils.hooks import collect_all, copy_metadata

SRC = os.path.abspath(os.path.join(SPECPATH, "..", "src"))
MODEL = os.path.join(SPECPATH, "model_bundle")

datas = [(MODEL, "hf_cache")]
binaries = []
hiddenimports = ["CompanyMatchingWithBERT", "ColumnMatchingWithBERTPhonetic",
                 "jellyfish", "openpyxl", "xlrd", "faiss"]
for pkg in ["sentence_transformers", "transformers", "tokenizers", "safetensors", "huggingface_hub"]:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
for dist in ["sentence-transformers", "transformers", "tokenizers", "safetensors", "huggingface-hub",
             "torch", "numpy", "tqdm", "regex", "requests", "httpx", "filelock", "packaging", "pyyaml"]:
    try:
        datas += copy_metadata(dist)
    except Exception:  # not installed with this version set; nothing to copy
        print(f"spec: no metadata for {dist}, skipped")

a = Analysis(
    [os.path.join(SRC, "CompanyMatch.py")],
    pathex=[SRC],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["matplotlib", "IPython", "jupyter", "notebook", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="CompanyMatch", console=True)
coll = COLLECT(exe, a.binaries, a.datas, name="CompanyMatch")
