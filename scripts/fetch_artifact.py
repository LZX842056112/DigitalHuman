"""Download a publisher-verified runtime artifact with resumable ranges."""
import argparse
from pathlib import Path
from scripts.download_modelscope import fetch_file

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('url')
    p.add_argument('destination',type=Path)
    p.add_argument('sha256')
    p.add_argument('size',type=int)
    a=p.parse_args()
    fetch_file(a.url,a.destination,a.sha256,a.size)
