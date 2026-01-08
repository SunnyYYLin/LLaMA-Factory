from os import getenv
from pathlib import Path
from random import sample
import heapq

import numpy as np
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
import pandas as pd
from llamafactory import data
from tap import Tap

"""ABC-preferences Dataset Example:
{
    'gene_seq': str,      # Unique gene sequence
    'cre_seqs': list[str],       # CRE sequences from all records whose TargetGene matches gene_seq
    'abc_scores': list[float],  # ABC scores corresponding to each CRE
    'activity': list[float],    # Activity values for each CRE
    'hic': list[float],         # Hi-C contact values for each CRE
    'gene_id': str,       # Gene identifier
    'gene_chrom': str,    # Chromosome of the gene
    'cre_pos': list[str], # Positions of each CRE in "chr:start-end" format
}
"""

def pickup_pairs(record: dict, 
                 min_diff: float = 0.1, 
                 max_num: int = -1) -> dict[str, list]:
    """Pick up pairs of CREs with significant activity differences."""
    cre_seqs = record['cre_seqs']
    abc_scores = record['abc_scores']
    cre_pos = record['cre_pos']
    
    # Sort activities by value (descending) using numpy for speed
    # numpy.argsort is significantly faster than python's sorted(enumerate(...)) for large lists
    acts_arr = np.array(abc_scores)
    sorted_inds = np.argsort(acts_arr)[::-1]
    sorted_vals = acts_arr[sorted_inds]
    
    n = len(sorted_vals)
    assert n >= 2, f"At least two CREs are required to form pairs, but got {n}."

    # Use a heap to search for max differences
    # Heap stores (-diff_val, i, j) where i, j are indices in sorted_vals
    heap = []
    
    # Start with the maximum possible difference
    val_i = sorted_vals[0]
    val_j = sorted_vals[n-1]
    diff_val = val_i - val_j
    
    if diff_val >= min_diff:
        heapq.heappush(heap, (-float(diff_val), 0, n-1))
    
    visited = {(0, n-1)}
    chosen_cres: list[str] = []
    rejected_cres: list[str] = []
    pair_poses: list[tuple[str, str]] = []
    diffs: list[float] = []
    
    while heap and (max_num < 0 or len(chosen_cres) < max_num):
        neg_d, i, j = heapq.heappop(heap)
        
        # Add to result
        chosen_cres.append(cre_seqs[sorted_inds[i]])
        rejected_cres.append(cre_seqs[sorted_inds[j]])
        diffs.append(-neg_d)
        pair_poses.append((cre_pos[sorted_inds[i]], cre_pos[sorted_inds[j]]))
        
        # Add neighbors: (i, j-1) and (i+1, j)
        
        # Neighbor 1: Decrease j
        if j - 1 > i:
            if (i, j-1) not in visited:
                d = sorted_vals[i] - sorted_vals[j-1]
                if d >= min_diff:
                    heapq.heappush(heap, (-float(d), i, j-1))
                    visited.add((i, j-1))
                    
        # Neighbor 2: Increase i
        if i + 1 < j:
            if (i+1, j) not in visited:
                d = sorted_vals[i+1] - sorted_vals[j]
                if d >= min_diff:
                    heapq.heappush(heap, (-float(d), i+1, j))
                    visited.add((i+1, j))
                    
    return {
        'gene_seq': record['gene_seq'],
        'chosen': chosen_cres, 
        'rejected': rejected_cres,
        'diff': diffs,
        'pair_pos': pair_poses,
    }

if __name__ == "__main__":
    class Args(Tap):
        dataset_dir: Path = Path('SunnyLin/ABC-preferences-K562')
        sample_rate: float = 1.0
        output_dir: Path|None = None
        max_num: int = 16
        min_diff: float = 0.1
    args = Args().parse_args()
    if not args.output_dir:
        args.output_dir = Path(getenv('DATASETS', '')) / 'lf' / args.dataset_dir.name
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        dataset = load_dataset(str(args.dataset_dir))
    except Exception as e:
        dataset = load_from_disk(args.dataset_dir)

    assert isinstance(dataset, DatasetDict), f"Expected a DatasetDict, but got {type(dataset)}"

    def process_batch(batch, min_diff, max_num):
        out = {
            'gene_seq': [],
            'chosen': [],
            'rejected': [],
            'diff': [],
            'pair_pos': [],
        }
        for i in range(len(batch['gene_seq'])):
            record = {k: batch[k][i] for k in batch.keys()}
            res = pickup_pairs(record, min_diff, max_num)
            for j in range(len(res['chosen'])):
                out['gene_seq'].append(res['gene_seq'])
                out['chosen'].append(res['chosen'][j])
                out['rejected'].append(res['rejected'][j])
                out['diff'].append(res['diff'][j])
                out['pair_pos'].append(res['pair_pos'][j])
        return out

    for split, subset in dataset.items():
        if args.sample_rate < 1.0:
            subset = subset.shuffle(seed=42).select(range(int(len(subset) * args.sample_rate)))
        print(f"Processing split '{split}' with {len(subset)} records.")
        subset = subset.map(
            process_batch,
            batched=True,
            remove_columns=subset.column_names,
            fn_kwargs={'min_diff': args.min_diff, 'max_num': args.max_num}
        )

        save_path = args.output_dir / f"{split}.json"
        print(f"Saving {len(subset)} records to {save_path}...")
        subset.to_json(save_path, force_ascii=False)

