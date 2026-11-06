"""Copy the all-MiniLM-L6-v2 model into ./model_bundle so the program can run offline.

Uses the local Hugging Face cache when the pinned revision is already there; otherwise
downloads that revision once. Writes only inside ./model_bundle next to this script.
"""
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"  # weights identical across the cached revisions
FILES = ["modules.json", "config_sentence_transformers.json", "sentence_bert_config.json", "README.md",
         "config.json", "model.safetensors", "special_tokens_map.json", "tokenizer.json",
         "tokenizer_config.json", "vocab.txt", "1_Pooling/config.json"]

here = Path(__file__).resolve().parent
out = here / "model_bundle" / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2"
if out.exists():
    shutil.rmtree(out)  # only ever removes the bundle this script itself created

src = Path(snapshot_download(REPO, revision=REVISION, allow_patterns=FILES))
dst = out / "snapshots" / REVISION
for name in FILES:
    target = dst / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / name, target)  # follows cache symlinks, so the bundle holds real files
(out / "refs").mkdir(parents=True, exist_ok=True)
(out / "refs" / "main").write_text(REVISION)
print(f"Model bundled at {out}")
