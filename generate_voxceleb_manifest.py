#!/usr/bin/env python3
import os
import csv

# Root VoxCeleb directory
ROOT_DIR = "/data/veer/testing/new/VoxCeleb1_test"
OUT_MANIFEST = "voxceleb_manifest.csv"

def make_manifest(root_dir, out_path):
    rows = []
    for speaker in os.listdir(root_dir):
        spk_dir = os.path.join(root_dir, speaker)
        if not os.path.isdir(spk_dir):
            continue
        for session in os.listdir(spk_dir):
            sess_dir = os.path.join(spk_dir, session)
            chunk_dir = os.path.join(sess_dir, "chunk_videos")
            if not os.path.isdir(chunk_dir):
                continue
            for wavfile in os.listdir(chunk_dir):
                if not wavfile.endswith(".wav"):
                    continue
                wav_path = os.path.join(chunk_dir, wavfile)
                segment_id = os.path.splitext(wavfile)[0]   # e.g., "5r0dWxy17C8#00001#594-803"
                rows.append([
                    segment_id,     # segment_id
                    wav_path,       # wav_path
                    "",             # start (empty = default 0.0)
                    "",             # end (empty = None)
                    speaker,        # speaker_id (e.g., id10270)
                    session         # session_id (e.g., YouTube video ID)
                ])

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    print(f"Manifest written with {len(rows)} entries to {out_path}")

if __name__ == "__main__":
    make_manifest(ROOT_DIR, OUT_MANIFEST)
