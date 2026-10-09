"""
main.py
-------
Entry point for the Python port of the CloudTracker satellite image application.

Replaces: src/main.cpp
- Windows native file dialog (GetOpenFileName / commdlg.h) →  tkinter.filedialog
- CLI interaction is identical to the C++ version.
"""

import sys
import tkinter as tk
from tkinter import filedialog

# Allow running from both the python/ directory and the project root
try:
    from cloud_tracker import CloudTracker
except ModuleNotFoundError:
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from cloud_tracker import CloudTracker


def open_file_dialog(title: str) -> str:
    """
    Show a cross-platform file-picker dialog and return the selected path.

    Replaces the Windows-only GetOpenFileNameA() call in main.cpp.
    Returns an empty string if the user cancels.
    """
    root = tk.Tk()
    root.withdraw()           # hide the empty root window
    root.attributes("-topmost", True)   # bring dialog to front
    path = filedialog.askopenfilename(
        title=title,
        filetypes=[
            ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
            ("All Files",   "*.*"),
        ],
    )
    root.destroy()
    return path or ""


def main() -> int:
    """
    Application entry point.

    1. Open file-picker dialogs to select T1 and T2 satellite images.
    2. Prompt for the time difference between passes.
    3. Run the CloudTracker pipeline.
    4. Display results in OpenCV windows.

    Returns
    -------
    int  0 on success, 1 on failure.
    """
    print("=== Satellite Image Cloud Tracker ===")

    print("Please select the first satellite image (T1)...")
    path1 = open_file_dialog("Select First Image (Time T1)")
    if not path1:
        print("No file selected. Exiting.")
        return 1

    print("Please select the second satellite image (T2)...")
    path2 = open_file_dialog("Select Second Image (Time T2)")
    if not path2:
        print("No file selected. Exiting.")
        return 1

    print(f"\nImages selected:")
    print(f"  T1: {path1}")
    print(f"  T2: {path2}")

    # Default to 1.0 hour if the user enters an invalid value
    time_diff = 1.0
    try:
        time_diff = float(
            input("\nEnter time difference between images (e.g., 1.5 hours): ")
        )
    except (ValueError, EOFError):
        print("Invalid input — using default of 1.0 hour.")

    tracker = CloudTracker()

    if tracker.load_images(path1, path2):
        print("Processing images... Please wait.")
        tracker.process(time_diff)
        print("Processing complete. Displaying results.")
        tracker.show_results()
    else:
        print("Failed to process images.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
