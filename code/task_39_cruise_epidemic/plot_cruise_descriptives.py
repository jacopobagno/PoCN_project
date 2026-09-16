from __future__ import annotations
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nodes', type=Path, required=True)
    parser.add_argument('--temporal-edges', type=Path, required=True)
    parser.add_argument('--aggregated-edges', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('results/cruise_epidemic_corrected'))
    return parser.parse_args()

def log_binned_density(values: np.ndarray, bins: int=24) -> tuple[np.ndarray, np.ndarray]:
    positive = np.asarray(values, dtype=float)
    positive = positive[positive > 0]
    edges = np.geomspace(positive.min(), positive.max() * (1.0 + 1e-09), bins + 1)
    counts, edges = np.histogram(positive, bins=edges)
    widths = np.diff(edges)
    centres = np.sqrt(edges[:-1] * edges[1:])
    present = counts > 0
    return (centres[present], counts[present] / (len(positive) * widths[present]))

def aggregated_degree_and_strength(nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ids = nodes['nodeID'].astype(str).to_numpy()
    index = {node_id: position for position, node_id in enumerate(ids)}
    left = edges['nodeID_from'].astype(str).map(index).to_numpy(dtype=int)
    right = edges['nodeID_to'].astype(str).map(index).to_numpy(dtype=int)
    weights = edges['weight'].to_numpy(dtype=float)
    degree = np.bincount(np.concatenate((left, right)), minlength=len(ids))
    strength = np.bincount(np.concatenate((left, right)), weights=np.concatenate((weights, weights)), minlength=len(ids))
    return (degree, strength)

def plot_descriptives(degree: np.ndarray, strength: np.ndarray, daily: pd.DataFrame, output: Path) -> None:
    degree_x, degree_y = log_binned_density(degree)
    strength_x, strength_y = log_binned_density(strength)
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.7))
    axes[0].loglog(degree_x, degree_y, marker='o', linestyle='None', color='#870011', markersize=3.4)
    axes[0].set(title='Degree distribution', xlabel='degree $k$', ylabel='density $P(k)$')
    axes[0].grid(alpha=0.2, which='both')
    axes[1].loglog(strength_x, strength_y, 'o', color='#d95f02', markersize=3.4)
    axes[1].set(title='Strength distribution', xlabel='strength $s$ (seconds)', ylabel='density $P(s)$')
    axes[1].grid(alpha=0.2, which='both')
    axis = axes[2]
    days = daily['time'].to_numpy(dtype=int)
    bars = axis.bar(days, daily['active_pairs'], width=0.55, color='#4477aa', label='active pairs')
    axis.set(title='Temporal activity', xlabel='observed day', ylabel='active pairs', xticks=days)
    axis.xaxis.set_label_coords(0.25, -0.19)
    axis.grid(alpha=0.2, axis='y')
    duration_axis = axis.twinx()
    duration_axis.plot(days, daily['total_duration_hours'], 'o', linestyle='None', color='#d95f02', markersize=4, label='contact duration')
    duration_axis.set_ylabel('total duration (hours)')
    axis.legend([bars, duration_axis.lines[0]], ['active pairs', 'contact duration'], loc='center', bbox_to_anchor=(0.8, -0.19), fontsize=7, frameon=False)
    fig.tight_layout(w_pad=2.0, rect=(0, 0.09, 1, 1))
    fig.savefig(output, dpi=300)
    plt.close(fig)

def main() -> None:
    args = arguments()
    nodes = pd.read_csv(args.nodes, dtype={'nodeID': str})
    temporal = pd.read_csv(args.temporal_edges, dtype={'nodeID_from': str, 'nodeID_to': str})
    aggregated = pd.read_csv(args.aggregated_edges, dtype={'nodeID_from': str, 'nodeID_to': str})
    degree, strength = aggregated_degree_and_strength(nodes, aggregated)
    daily = temporal.groupby('time', as_index=False).agg(active_pairs=('contact_duration', 'size'), total_duration_seconds=('contact_duration', 'sum')).sort_values('time')
    daily['total_duration_hours'] = daily['total_duration_seconds'] / 3600.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_descriptives(degree, strength, daily, args.output_dir / 'empirical_network_descriptives.png')
    with (args.output_dir / 'empirical_daily_activity.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(daily.columns))
        writer.writeheader()
        writer.writerows(daily.to_dict(orient='records'))
    print(f"Wrote {args.output_dir / 'empirical_network_descriptives.png'}")
if __name__ == '__main__':
    main()
