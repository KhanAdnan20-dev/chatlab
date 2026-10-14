import zipfile
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
zip_out = ROOT / "ChatLab-Colab-Project.zip"

include_dirs = ["backend", "frontend", "data", "scripts", "tests", "docs", "colab"]
include_files = [
    "README.md",
    ".gitignore",
    "requirements.txt",
    "requirements-gpu.txt",
]

exclude_patterns = [
    ".venv",
    "node_modules",
    "__pycache__",
    ".git",
    ".pytest_cache",
    "unsloth_compiled_cache",
    "models/qwen3-4b-bnb-4bit",
    "models/all-MiniLM-L6-v2",
    "models/qwen3-4b-qlora-adapter",
    ".log",
    ".pyc",
    "ChatLab-Colab-Project.zip",
]

def should_exclude(rel_path_str):
    rel_str = rel_path_str.replace("\\", "/")
    for exc in exclude_patterns:
        if exc in rel_str:
            return True
    return False

print("Building ChatLab-Colab-Project.zip...")
with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zipf:
    for fname in include_files:
        fp = ROOT / fname
        if fp.exists():
            zipf.write(fp, fname)
            print(f"  Added file: {fname}")

    for dname in include_dirs:
        dp = ROOT / dname
        if dp.exists():
            for p in dp.rglob("*"):
                if p.is_file():
                    rel_p = p.relative_to(ROOT)
                    rel_p_str = str(rel_p)
                    if not should_exclude(rel_p_str):
                        zipf.write(p, rel_p_str)

size_mb = zip_out.stat().st_size / (1024 * 1024)
print(f"[OK] Created {zip_out.name} ({size_mb:.2f} MB)")
