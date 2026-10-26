"""
Downloads and verifies real human face images for preset driving scenarios:
1. Alert driver (eyes open)
2. Yawning person (mouth wide open)
3. Sleeping person (eyes closed)
"""

import os
import sys
import json
import urllib.request
import cv2
import numpy as np

# Ensure project root in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.detector import DriverDrowsinessDetector

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
}

def get_wikimedia_url(title):
    url = f"https://commons.wikimedia.org/w/api.php?action=query&titles={urllib.parse.quote(title)}&prop=imageinfo&iiprop=url&format=json"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        d = json.loads(resp.read().decode('utf-8'))
        for p in d['query']['pages'].values():
            if 'imageinfo' in p:
                return p['imageinfo'][0]['url']
    return None

def fetch_and_save(title, out_path, target_name):
    url = get_wikimedia_url(title)
    if not url:
        print(f"Failed to find URL for {title}")
        return False
    print(f"Downloading {target_name} from: {url}")
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
        arr = np.asarray(bytearray(data), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            print("Failed to decode image")
            return False
        
        # Resize to standard HD frame (e.g. 720p or 640x480) keeping aspect ratio
        h, w = img.shape[:2]
        if w > 1280 or h > 720:
            scale = min(1280 / w, 720 / h)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        cv2.imwrite(out_path, img)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        det = DriverDrowsinessDetector()
        res = det.process_frame(rgb)
        det.close()
        print(f"[{target_name}] -> Face: {res['face_detected']} | Status: {res['driver_status']} | EAR: {res['avg_ear']} | MAR: {res['mar']} | Pitch: {res['head_pitch_deg']}")
        return True

def run():
    out_dir = os.path.join(root_dir, "data", "sample_presets")
    os.makedirs(out_dir, exist_ok=True)

    print("Fetching real human photos for Driver Drowsiness Detection...\n")
    # 1. Alert Truck Driver
    fetch_and_save(
        "File:Romanian Truck Driver (62829107).jpeg",
        os.path.join(out_dir, "driver_alert_1.jpg"),
        "Real Alert Truck Driver"
    )

    # 2. Yawning Person
    fetch_and_save(
        "File:Portrait of young man, yawning (4419926703).jpg",
        os.path.join(out_dir, "driver_yawn_1.jpg"),
        "Real Yawning Driver"
    )

    # 3. Sleeping Person (Closed eyes)
    fetch_and_save(
        "File:Person_Sleeping.JPG",
        os.path.join(out_dir, "driver_microsleep_1.jpg"),
        "Real Sleeping Person"
    )

if __name__ == "__main__":
    run()
