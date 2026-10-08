"""Package source and documentation without environments, caches or weights."""
from pathlib import Path
import zipfile

root=Path(__file__).resolve().parents[1]
target=root / "RL_project_source.zip"
if target.exists():
    raise FileExistsError(f"Preserving existing archive: {target}; rename it before packaging again")
files=[]
for folder in ("racing","tests","scripts","docs"):
    files.extend(p for p in (root/folder).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix!=".pyc")
files.extend(root/name for name in ("README.md","requirements.txt","requirements-resolved-local.txt",
                                   "课上讲过的算法.md",".gitignore") if (root/name).exists())
with zipfile.ZipFile(target,"x",zipfile.ZIP_DEFLATED) as archive:
    for file in sorted(files):
        archive.write(file,file.relative_to(root).as_posix())
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
print(f"Created {target}, {len(files)} files, {target.stat().st_size} bytes")
