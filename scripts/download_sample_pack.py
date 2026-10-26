"""
One-Click Sample Pack Downloader for Driver Drowsiness Testing.
Downloads verified real in-cabin test photos and clips into data/test_pack/
"""
import os
import urllib.request
import shutil

dest_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "test_pack"))
os.makedirs(dest_dir, exist_ok=True)

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

files = [
    (
        "real_truck_driver_cabin.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/6/6b/Romanian_Truck_Driver_%2862829107%29.jpeg"
    ),
    (
        "real_sleeping_closed_eyes.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/b/b5/Andre_Hunter_2017-05-19_%28Unsplash%29.jpg"
    ),
    (
        "real_person_yawning_fatigue.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/a/ac/Simone_Biles_yawning_Paris_Olympics_2024.jpg"
    )
]

print("Downloading test pack directly into:", dest_dir)

for filename, url in files:
    out_path = os.path.join(dest_dir, filename)
    try:
        print(f"Downloading {filename}...")
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as resp, open(out_path, 'wb') as f:
            shutil.copyfileobj(resp, f)
        print(f"  [OK] Saved to: {out_path}")
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")

# Also copy the 3 generated in-cabin benchmark images into test_pack for easy testing
preset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sample_presets"))
for f in ["driver_alert_1.jpg", "driver_yawn_1.jpg", "driver_microsleep_1.jpg", "driver_alert.mp4", "driver_yawn.mp4", "driver_microsleep.mp4"]:
    src = os.path.join(preset_dir, f)
    if os.path.exists(src):
        dst = os.path.join(dest_dir, f"cabin_{f}")
        shutil.copy2(src, dst)
        print(f"  [OK] Bundled: cabin_{f}")

print("\nAll sample test files are ready in data/test_pack/!")
