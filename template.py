import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")

folder_files_list = [
    ".github/workflows/.gitkeep",
    "src/__init__.py",
    "src/components/__init__.py",
    # "src/utils/__init__.py",
    # "src/config/__init__.py",
    # "src/config/configuration.py",
    "src/pipeline/__init__.py",
    "src/entity/__init__.py",
    "src/constants/__init__.py",
    # "requirements.txt",
    # "setup.py",
    "research/01_Vcc_Trials.ipynb",
    "templates/index.html",
    "artifacts/Data/delete.csv",
    "artifacts/DataFrame/delete.csv",
    "artifacts/Models/delete.csv",
    "artifacts/Scores/delete.csv",
    "artifacts/Pipeline/delete.csv",
]

for path in folder_files_list:
    file_path = Path(path)
    file_dir, file_name = os.path.split(file_path)

    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f'Created directory "{file_dir}" for file "{file_name}"')

    if (not os.path.exists(path)) or (os.path.getsize(path) == 0):
        with open(path, "w") as f:
            pass
        logging.info(f'Created empty file: "{path}"')
    else:
        logging.info(f'"{path}" already exists')
