#!/usr/bin/env python3
"""
ImmunePlan Clean-up Script
Removes all generated models, checkpoints, logs, data, and temporary files.
Will NOT remove code, .env file, or benchmark/source JSONs tracked in git.
"""

import os
import shutil
from pathlib import Path
import argparse

# Add huggingface cache to the list of absolute paths to wipe
HF_CACHE = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface/hub'))

# Directories to completely remove
DIRS_TO_REMOVE = [
    "models",
    "checkpoints",
    "logs",
    "results",
    "data",  # The data dir is generated (pdfs, vector_store, raw_papers.json)
    "wandb",
    "tensorboard",
    ".pytest_cache",
    HF_CACHE
]

# Specific files to remove in the project root
FILES_TO_REMOVE = [
    "model_info.json",
    ".DS_Store"
]

def format_size(bytes_size):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0

def get_dir_size(path):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total

def run_cleanup(dry_run=False, auto_confirm=False):
    project_root = Path(__file__).parent.resolve()
    
    targets_to_delete = []
    total_size_bytes = 0
    
    # Find exact paths to remove (handle absolute paths for things outside project root)
    for d in DIRS_TO_REMOVE:
        p = Path(d)
        if not p.is_absolute():
            p = project_root / d
            
        if p.exists() and p.is_dir():
            size = get_dir_size(p)
            targets_to_delete.append((p, "dir", size))
            total_size_bytes += size
            
    for f in FILES_TO_REMOVE:
        p = project_root / f
        if p.exists() and p.is_file():
            size = p.stat().st_size
            targets_to_delete.append((p, "file", size))
            total_size_bytes += size
            
    # Find all __pycache__ and .pyc
    for path in project_root.rglob("__pycache__"):
        if path.is_dir():
            size = get_dir_size(path)
            targets_to_delete.append((path, "dir", size))
            total_size_bytes += size
            
    for path in project_root.rglob("*.pyc"):
        if path.is_file():
            size = path.stat().st_size
            targets_to_delete.append((path, "file", size))
            total_size_bytes += size

    # Remove duplicates from rglob just in case
    unique_targets = {t[0]: t for t in targets_to_delete}.values()

    if not unique_targets:
        print("Everything is already clean! Nothing to remove.")
        return

    print("The following files and directories will be REMOVED:")
    print("-" * 60)
    for path, t_type, size in sorted(unique_targets, key=lambda x: str(x[0])):
        # Attempt to make relative to project root, if possible, else use absolute
        try:
            display_path = str(path.relative_to(project_root))
        except ValueError:
            display_path = str(path)
            
        print(f"[{t_type.upper():>4}] {display_path:<40} ({format_size(size)})")
    print("-" * 60)
    print(f"Total space to free: {format_size(total_size_bytes)}")
    print("-" * 60)

    if dry_run:
        print("\nDry run mode active. No files were actually deleted.")
        return

    if not auto_confirm:
        confirm = input("\nAre you sure you want to delete these files? This cannot be undone. [y/N]: ")
        if confirm.lower() not in ['y', 'yes']:
            print("Cleanup aborted.")
            return

    print("\nCleaning up...")
    for path, t_type, _ in unique_targets:
        try:
            if t_type == "dir":
                shutil.rmtree(path)
            else:
                path.unlink()
        except Exception as e:
            try:
                display_path = str(path.relative_to(project_root))
            except ValueError:
                display_path = str(path)
            print(f"Failed to remove {display_path}: {e}")

    print("\nCleanup complete! Project is back to a clean state.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up generated pipeline files.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without actually deleting")
    parser.add_argument("-y", "--yes", action="store_true", help="Automatically confirm deletion")
    args = parser.parse_args()
    
    run_cleanup(dry_run=args.dry_run, auto_confirm=args.yes)
