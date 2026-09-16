from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
import networkx as nx
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt


MAROON = '#870011'


def load_graph(path: Path, n_nodes: int) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            fields = line.split()
            if fields:
                left, right = map(int, fields[:2])
                if left != right:
                    graph.add_edge(left, right)
    return graph


def degree_profiles(graph: nx.Graph) -> list[dict[str, float]]:
    degrees = dict(graph.degree())
    neighbour_degrees = nx.average_neighbor_degree(graph)
    local_clustering = nx.clustering(graph)
    grouped_neighbours: dict[int, list[float]] = defaultdict(list)
    grouped_clustering: dict[int, list[float]] = defaultdict(list)
    for node, degree in degrees.items():
        if degree > 0:
            grouped_neighbours[degree].append(float(neighbour_degrees[node]))
            grouped_clustering[degree].append(float(local_clustering[node]))
    return [
        {
            'degree': degree,
            'node_count': len(grouped_neighbours[degree]),
            'average_nearest_neighbor_degree': float(np.mean(grouped_neighbours[degree])),
            'average_local_clustering': float(np.mean(grouped_clustering[degree])),
        }
        for degree in sorted(grouped_neighbours)
    ]


def log_binned_profiles(rows: list[dict[str, float]], bins: int = 24) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    degree = np.asarray([row['degree'] for row in rows], dtype=float)
    neighbours = np.asarray([row['average_nearest_neighbor_degree'] for row in rows])
    clustering = np.asarray([row['average_local_clustering'] for row in rows])
    counts = np.asarray([row['node_count'] for row in rows])
    boundaries = np.geomspace(degree.min(), degree.max() + 1.0, bins + 1)
    indices = np.clip(np.digitize(degree, boundaries) - 1, 0, bins - 1)
    binned_degree, binned_neighbours, binned_clustering = [], [], []
    for index in range(bins):
        selected = indices == index
        if np.any(selected):
            weights = counts[selected]
            binned_degree.append(float(np.exp(np.average(np.log(degree[selected]), weights=weights))))
            binned_neighbours.append(float(np.average(neighbours[selected], weights=weights)))
            binned_clustering.append(float(np.average(clustering[selected], weights=weights)))
    return np.asarray(binned_degree), np.asarray(binned_neighbours), np.asarray(binned_clustering)


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_profiles(rows: list[dict[str, float]], output_dir: Path) -> None:
    degree, neighbours, clustering = log_binned_profiles(rows)
    figure, axis = plt.subplots(figsize=(3.1, 2.25))
    axis.loglog(degree, neighbours, 'o-', color=MAROON, markersize=3.2, linewidth=1.0)
    axis.set(title='Average nearest-neighbor degree vs degree k', xlabel='$k$', ylabel='$k_{nn}(k)$')
    axis.grid(alpha=0.2, which='both')
    figure.tight_layout()
    figure.savefig(output_dir / 'knn_by_degree.png', dpi=300)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(3.1, 2.25))
    axis.semilogx(degree, clustering, 'o-', color=MAROON, markersize=3.2, linewidth=1.0)
    axis.set(title='Average local clustering vs degree k', xlabel='$k$', ylabel='$c(k)$', ylim=(0, 1))
    axis.grid(alpha=0.2, which='both')
    figure.tight_layout()
    figure.savefig(output_dir / 'clustering_by_degree.png', dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('edge_list', type=Path)
    parser.add_argument('--n-nodes', type=int, default=5535)
    parser.add_argument('--output-dir', type=Path, default=Path('results/analysis_seed42'))
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    rows = degree_profiles(load_graph(arguments.edge_list, arguments.n_nodes))
    write_csv(arguments.output_dir / 'degree_correlations.csv', rows)
    plot_profiles(rows, arguments.output_dir)


if __name__ == '__main__':
    main()
