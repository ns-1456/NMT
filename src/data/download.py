"""Dataset download utilities for English-Gujarati parallel corpus"""

import os
import requests
from pathlib import Path
from typing import Tuple, Optional
from tqdm import tqdm
import zipfile
import tarfile


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> None:
    """Download a file from URL with progress bar.
    
    Args:
        url: URL to download from
        dest_path: Destination file path
        chunk_size: Chunk size for downloading
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def download_flores_dataset(data_dir: Path) -> Tuple[Path, Path]:
    """Download FLORES-200 English-Gujarati dataset.
    
    Args:
        data_dir: Directory to save the dataset
        
    Returns:
        Tuple of (english_file, gujarati_file) paths
    """
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # FLORES-200 devtest and dev sets
    base_url = "https://github.com/facebookresearch/flores/raw/main/flores200_dataset"
    
    files = {
        "en": {
            "dev": f"{base_url}/dev/dev.eng_Latn",
            "devtest": f"{base_url}/devtest/devtest.eng_Latn"
        },
        "gu": {
            "dev": f"{base_url}/dev/dev.guj_Gujr",
            "devtest": f"{base_url}/devtest/devtest.guj_Gujr"
        }
    }
    
    en_files = []
    gu_files = []
    
    for split in ["dev", "devtest"]:
        en_path = raw_dir / f"flores_{split}.en"
        gu_path = raw_dir / f"flores_{split}.gu"
        
        if not en_path.exists():
            print(f"Downloading FLORES {split} English...")
            download_file(files["en"][split], en_path)
        
        if not gu_path.exists():
            print(f"Downloading FLORES {split} Gujarati...")
            download_file(files["gu"][split], gu_path)
        
        en_files.append(en_path)
        gu_files.append(gu_path)
    
    # Combine dev and devtest
    combined_en = raw_dir / "flores_combined.en"
    combined_gu = raw_dir / "flores_combined.gu"
    
    if not combined_en.exists():
        with open(combined_en, 'w', encoding='utf-8') as f:
            for file in en_files:
                with open(file, 'r', encoding='utf-8') as src:
                    f.write(src.read())
    
    if not combined_gu.exists():
        with open(combined_gu, 'w', encoding='utf-8') as f:
            for file in gu_files:
                with open(file, 'r', encoding='utf-8') as src:
                    f.write(src.read())
    
    return combined_en, combined_gu


def download_opus_dataset(data_dir: Path, corpus_name: str = "GNOME") -> Optional[Tuple[Path, Path]]:
    """Download OPUS dataset for English-Gujarati.
    
    Note: OPUS datasets may require manual download or API access.
    This is a placeholder for the download logic.
    
    Args:
        data_dir: Directory to save the dataset
        corpus_name: Name of the OPUS corpus (e.g., "GNOME", "KDE", "Ubuntu")
        
    Returns:
        Tuple of (english_file, gujarati_file) paths or None if not available
    """
    # OPUS datasets typically require:
    # 1. Access through opus.nlpl.eu
    # 2. Or manual download from https://opus.nlpl.eu/
    # This function should be implemented based on specific corpus requirements
    
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"OPUS dataset download for {corpus_name} not yet implemented.")
    print("Please download manually from https://opus.nlpl.eu/")
    print("Or use the FLORES dataset which is automatically downloadable.")
    
    return None


def prepare_dataset(data_dir: Path, dataset_name: str = "flores") -> Tuple[Path, Path]:
    """Prepare English-Gujarati dataset.
    
    Args:
        data_dir: Directory to save the dataset
        dataset_name: Name of dataset to download ("flores" or "opus")
        
    Returns:
        Tuple of (english_file, gujarati_file) paths
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if dataset_name.lower() == "flores":
        return download_flores_dataset(data_dir)
    elif dataset_name.lower() == "opus":
        result = download_opus_dataset(data_dir)
        if result is None:
            # Fallback to FLORES
            print("Falling back to FLORES dataset...")
            return download_flores_dataset(data_dir)
        return result
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose 'flores' or 'opus'")
