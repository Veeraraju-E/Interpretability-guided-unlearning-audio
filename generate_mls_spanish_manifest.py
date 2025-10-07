#!/usr/bin/env python3
"""
Generate manifest for MLS Spanish dataset.
Similar to generate_voxceleb_manifest.py but adapted for MLS Spanish structure.
"""

import os
import csv
import glob
from pathlib import Path

def make_manifest(root_dir, out_path):
    """
    Generate manifest for MLS Spanish dataset.
    
    Args:
        root_dir: Path to MLS Spanish dev/audio directory
        out_path: Output CSV file path
    """
    rows = []
    
    # MLS Spanish structure: /dev/audio/{speaker_id}/{subfolder_id}/*.flac
    pattern = os.path.join(root_dir, "*", "*", "*.flac")
    audio_files = glob.glob(pattern)
    
    print(f"Found {len(audio_files)} audio files")
    
    for audio_file in sorted(audio_files):
        path_parts = Path(audio_file).parts
        if len(path_parts) >= 3:
            speaker_id = path_parts[-3]  # speaker folder name
            filename = os.path.basename(audio_file)
            
            # Create a unique segment ID
            segment_id = f"{speaker_id}_{filename}"
            
            # Add to manifest
            rows.append([
                segment_id,  # segment_id
                audio_file,  # path
                speaker_id,  # speaker_id (same as label)
                speaker_id,  # label
                speaker_id   # speaker (for compatibility)
            ])
    
    # Write CSV
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['segment_id', 'path', 'speaker_id', 'label', 'speaker'])
        writer.writerows(rows)
    
    print(f"Manifest written with {len(rows)} entries to {out_path}")
    
    # Print some statistics
    speakers = set(row[2] for row in rows)
    print(f"Found {len(speakers)} unique speakers: {sorted(speakers)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python generate_mls_spanish_manifest.py <root_dir> <out_path>")
        print("Example: python generate_mls_spanish_manifest.py /data/veer/testing/new/datasets/mls_spanish/dev/audio /data/veer/testing/new/datasets/mls_spanish/manifest.csv")
        sys.exit(1)
    
    root_dir = sys.argv[1]
    out_path = sys.argv[2]
    
    if not os.path.exists(root_dir):
        print(f"Error: Root directory {root_dir} does not exist")
        sys.exit(1)
    
    make_manifest(root_dir, out_path)
