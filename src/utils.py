import os
from typing import List
from pydantic import FilePath, DirectoryPath
from pathlib import Path


def traverse_directory_for_files(root_folder: DirectoryPath) -> List[FilePath]:
    output_files: List[FilePath] = []
    for dirpath, _, filenames in os.walk(root_folder):
        for file_name in filenames:
            file_path: FilePath = Path(dirpath) / file_name
            output_files.append(file_path)
    return output_files
