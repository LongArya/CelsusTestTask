import os
import json
from typing import List, Dict
from pydantic import FilePath, DirectoryPath
from pathlib import Path


def traverse_directory_for_files(root_folder: DirectoryPath) -> List[FilePath]:
    output_files: List[FilePath] = []
    for dirpath, _, filenames in os.walk(root_folder):
        for file_name in filenames:
            file_path: FilePath = Path(dirpath) / file_name
            output_files.append(file_path)
    return output_files


def write_json(data: Dict, json_file: Path) -> None:
    with open(json_file, "w") as json_file:
        json.dump(data, json_file, indent=2)


def read_json(json_file: FilePath) -> Dict:
    with open(json_file, "r") as json_file:
        data = json.load(json_file)
    return data
