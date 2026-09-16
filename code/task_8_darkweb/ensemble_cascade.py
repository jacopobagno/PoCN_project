from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from analyse_darknet import motter_lai_cascade, prepare_motter_lai_cascade
from generate_darknet import DEFAULT_EDGES, DEFAULT_NODES, generate_darknet
DEFAULT_ALPHAS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.75, 1.0)

def parse_seed_list(value: str) -> list[int]:
    return [int(seed) for seed in value.split(',')]

def build_graph(seed: int) -> nx.Graph:
    edges, _, _, _ = generate_darknet(n_nodes=DEFAULT_NODES, n_edges=DEFAULT_EDGES, n0=200, e0=1000, beta=1.0, gamma=1.0, bandwidth_mu=0.0, bandwidth_sigma=1.0, rewire_attempts=DEFAULT_EDGES, seed=seed)
    graph = nx.Graph()
    graph.add_nodes_from(range(DEFAULT_NODES))
    graph.add_edges_from(edges)
    return graph

def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def plot_mean(rows: list[dict], path: Path) -> None:
    alpha = [row['alpha'] for row in rows]
    lcc = [row['mean_final_lcc'] for row in rows]
    fig, axis = plt.subplots(figsize=(3.1, 2.25))
    axis.plot(alpha, lcc, 'o-', color='#870011', linewidth=1.4, markersize=3.2, label='ensemble mean')
    axis.axvline(0.2, color='#4a4a4a', linestyle='--', linewidth=0.9, label='2015 reference $\\alpha\\approx0.2$')
    axis.set(xlabel='$\\alpha$', ylabel='relative LCC size', xlim=(0, 1), ylim=(0, 1))
    axis.set_title('Cascade failures', fontsize=9)
    axis.legend(frameon=False, fontsize=5.8, loc='lower right')
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('results/analysis_ensemble'))
    parser.add_argument('--network-seeds', default='42,43,44')
    parser.add_argument('--cohort-seeds', default='101,202,303')
    parser.add_argument('--sources', type=int, default=128)
    parser.add_argument('--max-generations', type=int, default=100)
    return parser.parse_args()

def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    network_seeds = parse_seed_list(args.network_seeds)
    cohort_seeds = parse_seed_list(args.cohort_seeds)
    individual_rows: list[dict] = []
    for network_seed in network_seeds:
        print(f'Generating network seed {network_seed}...', flush=True)
        graph = build_graph(network_seed)
        for cohort_seed in cohort_seeds:
            print(f'  Cohort seed {cohort_seed}...', flush=True)
            prepared = prepare_motter_lai_cascade(graph, args.sources, cohort_seed)
            for alpha in DEFAULT_ALPHAS:
                result = motter_lai_cascade(graph, alpha, args.sources, cohort_seed, args.max_generations, prepared)
                individual_rows.append({'network_seed': network_seed, 'cohort_seed': cohort_seed, **result})
                print(f"    alpha={alpha:.2f}, final_LCC={result['largest_component_fraction']:.4f}", flush=True)
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in individual_rows:
        grouped[float(row['alpha'])].append(float(row['largest_component_fraction']))
    mean_rows = [{'alpha': alpha, 'mean_final_lcc': float(np.mean(grouped[alpha])), 'n_runs': len(grouped[alpha])} for alpha in DEFAULT_ALPHAS]
    write_rows(args.output_dir / 'cascade_ensemble_individual.csv', individual_rows)
    write_rows(args.output_dir / 'cascade_ensemble_mean.csv', mean_rows)
    plot_mean(mean_rows, args.output_dir / 'cascade_ensemble_mean.png')
    print('Ensemble cascade experiment complete.', flush=True)
if __name__ == '__main__':
    main()
