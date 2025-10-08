#!/usr/bin/env python3
import os
import csv

# Root MLS Spanish directory
ROOT_DIR = "/data/veer/testing/new/datasets/mls_spanish/dev/audio"
OUT_MANIFEST = "mls_spanish_manifest.csv"

def make_manifest(root_dir, out_path):
    rows = []
    for speaker in os.listdir(root_dir):
        spk_dir = os.path.join(root_dir, speaker)
        if not os.path.isdir(spk_dir):
            continue
        for book in os.listdir(spk_dir):
            book_dir = os.path.join(spk_dir, book)
            if not os.path.isdir(book_dir):
                continue
            for flacfile in os.listdir(book_dir):
                if not flacfile.endswith(".flac"):
                    continue
                flac_path = os.path.join(book_dir, flacfile)
                segment_id = os.path.splitext(flacfile)[0]   # e.g., "10367_10282_000000"
                rows.append([
                    segment_id,     # segment_id
                    flac_path,      # flac_path
                    "",             # start (empty = default 0.0)
                    "",             # end (empty = None)
                    speaker,        # speaker_id (e.g., 10367)
                    book            # book_id (e.g., 10282)
                ])

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    print(f"Manifest written with {len(rows)} entries to {out_path}")

if __name__ == "__main__":
    make_manifest(ROOT_DIR, OUT_MANIFEST)
