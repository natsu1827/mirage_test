from pathlib import Path
import requests

def download(url, save_dir="__image"):
    Path(save_dir).mkdir(exist_ok=True)
    path = Path(save_dir) / "input.jpg"
    r = requests.get(url)
    r.raise_for_status()
    path.write_bytes(r.content)
    return path