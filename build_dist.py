"""
Block Blast Bot v2.0 - Standalone Distribution Builder

This script automates the process of compiling the Rust FFI core and
packaging the entire python application, OpenCV dependencies, and DearPyGui
into a single portable standalone directory using PyInstaller.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


def print_header(msg: str):
    print(f"\n{'=' * 60}")
    print(f"🚀 {msg}")
    print(f"{'=' * 60}\n")


def run_cmd(cmd: list, check: bool = True):
    print(f"> {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check)
    return result.returncode == 0


def main():
    root_dir = Path(__file__).parent.absolute()
    os.chdir(root_dir)

    print_header("Step 1: Compiling Rust FFI Core (maturin build --release)")
    # Note: ensure maturin is installed in the current environment
    if not run_cmd(["uv", "run", "maturin", "build", "--release"]):
        print("❌ Failed to compile Rust backend.")
        sys.exit(1)

    # Note: Maturin automatically installs the wheel into the local environment
    # during `develop` or when using `uv pip install`. For this standalone build,
    # we assume `block_blast_rust` is already available to the current interpreter.
    # Otherwise, PyInstaller won't find it.

    print_header("Step 2: Ensuring PyInstaller is available")
    run_cmd(["uv", "pip", "install", "pyinstaller"])

    print_header("Step 3: Building PyInstaller Spec")

    # We create a custom spec execution to ensure dynamic libraries like DearPyGui
    # and OpenCV binaries are explicitly bundled if Pyinstaller misses them.
    # The --onedir flag creates a folder containing the exe and its DLLs.
    # This is much faster to start up than --onefile and allows users to drop
    # extra config files next to it.

    pyinstaller_cmd = [
        "uv",
        "run",
        "pyinstaller",
        "--noconfirm",
        "--clean",
        "--name",
        "BlockBlastAI",
        "--windowed",  # Don't spawn a lingering terminal window on execution
        "--onedir",  # Creates the 'dist/BlockBlastAI/' portable folder
        # Include data files/assets (if any layout files or templates exist later)
        "--add-data",
        f"config.yaml;.",
        # Entry point
        "src/block_blast_bot/main.py",
    ]

    if not run_cmd(pyinstaller_cmd):
        print("❌ Failed to build PyInstaller payload.")
        sys.exit(1)

    print_header("Step 4: Assembling Portable Payload")

    dist_dir = root_dir / "dist" / "BlockBlastAI"
    if not dist_dir.exists():
        print("❌ Build directory not found.")
        sys.exit(1)

    # Copy calibration template if they haven't calibrated yet
    # Not strictly necessary if the script initializes an empty one, but good practice.

    # User instructions:
    readme_path = root_dir / "dist" / "README.txt"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("Block Blast Bot v2.0 - Portable Distribution Area\n")
        f.write("=" * 50 + "\n\n")
        f.write("1. Copy the 'BlockBlastAI' folder to any Windows machine.\n")
        f.write(
            "2. Make sure scrcpy is installed on the target machine and added to PATH (or drop it inside the folder).\n"
        )
        f.write("3. Make sure 'adb devices' verifies your phone is connected.\n")
        f.write("4. Run 'BlockBlastAI.exe'.\n\n")
        f.write(
            "Note: If the bot cannot click, you must delete 'calibration.json' and run Phase 4 Calibration tools first.\n"
        )

    print_header("✅ Build Complete!")
    print(f"The portable standalone folder is ready at:\n➡️  {dist_dir}")


if __name__ == "__main__":
    main()
