from pathlib import Path
import sys

K, N, L = 4, 8, 2  # rows, columns, ones per column

HERE = Path(__file__).resolve().parent
PEG_ROOT = HERE.parents[1] / "ldpc-for-flash-memory"
if not (PEG_ROOT / "PEG" / "PEG.py").is_file():
    raise FileNotFoundError(f"PEG submodule not found: {PEG_ROOT}")

sys.path.insert(0, str(PEG_ROOT))
from PEG.PEG import PEG

OUTPUT = HERE / "data" / f"peg_{K}x{N}_l{L}.txt"
H = PEG(K, N, L).peg(OUTPUT, delimiter=" ")
