import shutil
from pathlib import Path
from typing import List, Tuple


def get_index_folders_and_files(index_folder_path: Path) -> Tuple[List[Path], List[Path]]:
    """Paths of the subfolders and files of the index."""
    folders = [ 
        index_folder_path / 'index',
        index_folder_path / 'residue_data_folder' # This is where residue data, extracted from parsing PDB files, is saved as a pickled and compressed file.
    ]
    files = [
        index_folder_path / 'indexed_PDB_files.txt',
    ]
    return (folders, files)

def index_folder_exists(index_folder_path: Path) -> bool:
    """Checks the index folder, its subfolders and files all exist."""
    folders, files = get_index_folders_and_files(index_folder_path)
    return index_folder_path.exists() and all(file.is_file() for file in files) and all(folder.is_dir() for folder in folders)

def create_index_folder_tree(index_folder_path: Path) -> None:
    """Creates the index folder as well as its subfolders and files."""
    if index_folder_path.exists():
        shutil.rmtree(index_folder_path, ignore_errors=True)

    index_folder_path.mkdir()

    folders, files = get_index_folders_and_files(index_folder_path)
    for folder in folders:
        folder.mkdir()
    for file in files:
        file.touch() # Creates an empty file

    return
