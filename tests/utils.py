from pathlib import Path

# Where to find tests data
DATA_FOLDER = Path(__file__).parent / "data"
assert DATA_FOLDER.is_dir()

# Where to store tests outputs
OUTPUT_FOLDER = Path(__file__).parent / "output"
DATA_FOLDER.mkdir(exist_ok=True)
