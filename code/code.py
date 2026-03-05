import json
import os

with open(os.getenv("QUARTO_EXECUTE_INFO")) as f:
    execute_info = json.load(f)
execute_info["document-path"]

