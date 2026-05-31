import sys
import os

exe_path = "CloudTracker.exe"
if not os.path.exists(exe_path):
    print("Executable not found!")
    sys.exit(1)

with open(exe_path, "rb") as f:
    content = f.read()

# Replace " units/time" with " km/hr\0\0\0\0\0\0" (same length)
target_str = b" units/time"
replacement_str = b" km/hr     "
if target_str in content:
    content = content.replace(target_str, replacement_str)
    print("Patched console string!")
else:
    print("Console string not found.")

# Try replacing "Wind Speed" if it exists with padding
target_img = b"Wind Speed\0"
replacement_img = b"Wind Speed\0"
# Actually, patching image text is risky if we don't have enough space. Let's just patch the console string first.

with open(exe_path, "wb") as f:
    f.write(content)

print("Patching complete.")
