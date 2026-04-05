from pathlib import Path
import sys


SRC_PATH = Path(__file__).resolve().parent / "src"
src_path_str = str(SRC_PATH)

if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)
