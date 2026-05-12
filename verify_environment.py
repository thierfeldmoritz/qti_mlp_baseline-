import importlib
import sys

modules = [
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "nibabel",
    "torch",
    "torchmetrics",
    "tensorboard",
    "cmasher",
]

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version.split()[0]}")

failed = []
for name in modules:
    try:
        importlib.import_module(name)
        print(f"[OK] {name}")
    except Exception as exc:
        print(f"[FAIL] {name}: {exc}")
        failed.append(name)

if failed:
    raise SystemExit(f"\nMissing/broken packages: {', '.join(failed)}")

print("\nEnvironment check passed.")
