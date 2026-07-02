"""
save_norm_stats.py
------------------
Compute and save per-criterion z-score normalisation statistics from the
BioModule v3 training subjects (S1, S5, S6, S7, S8).

Run this once after setting up the aligned data to produce
checkpoint/norm_stats.json, which is required for training and inference.
The pre-computed file is already committed at checkpoint/norm_stats.json
for convenience — only re-run if you retrain on a different data split.

Usage
-----
  python scripts/save_norm_stats.py \\
      --db-root    /path/to/h36m_database/aligned_data \\
      --out        checkpoint/norm_stats.json
"""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bio_module_v3.dataset import compute_norm_stats_v2

TRAIN_SUBJECTS = ['S1', 'S5', 'S6', 'S7', 'S8']


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db-root', required=True,
                   help='Root of the aligned data (contains S1/, S5/, …)')
    p.add_argument('--out', default='checkpoint/norm_stats.json',
                   help='Output path for norm_stats.json')
    p.add_argument('--subjects', nargs='+', default=TRAIN_SUBJECTS,
                   help='Subjects to use for statistics (default: S1 S5 S6 S7 S8)')
    args = p.parse_args()

    print(f'Computing norm stats from subjects: {args.subjects}')
    norm_stats = compute_norm_stats_v2(args.db_root, args.subjects)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(norm_stats, f, indent=2)

    n_crit = len(norm_stats.get('mean', {}))
    print(f'Saved {n_crit} criteria → {out_path}')


if __name__ == '__main__':
    main()
