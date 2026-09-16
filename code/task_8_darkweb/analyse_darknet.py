from __future__ import annotations
import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path
import matplotlib
import networkx as nx
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from generate_darknet import degree_preserving_rewire
EMPIRICAL_2015 = {'nodes': 5535, 'edges': 274831, 'mean_degree': 99.3066, 'global_clustering': 'high (exact value not stated in the article text)', 'average_path_length': '2.0--2.5', 'diameter': '4--5', 'assortativity': 'negative'}

def load_graph(edge_list: Path, n_nodes: int) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    with edge_list.open(encoding='utf-8') as file:
        for line in file:
            fields = line.split()
            if not fields:
                continue
            u, v = map(int, fields[:2])
            if u != v:
                graph.add_edge(u, v)
    return graph

def largest_component(graph: nx.Graph) -> nx.Graph:
    nodes = max(nx.connected_components(graph), key=len)
    return graph.subgraph(nodes).copy()

def sampled_path_metrics(graph: nx.Graph, n_sources: int, seed: int) -> tuple[float, int]:
    rng = np.random.default_rng(seed)
    sources = rng.choice(list(graph), size=min(n_sources, graph.number_of_nodes()), replace=False)
    distances: list[int] = []
    sampled_diameter = 0
    for source in sources:
        lengths = nx.single_source_shortest_path_length(graph, int(source))
        values = list(lengths.values())
        distances.extend((distance for distance in values if distance > 0))
        sampled_diameter = max(sampled_diameter, max(values, default=0))
    return (float(np.mean(distances)), sampled_diameter)

def structural_metrics(graph: nx.Graph, path_sources: int, seed: int) -> tuple[dict, nx.Graph]:
    lcc = largest_component(graph)
    degrees = np.fromiter((degree for _, degree in graph.degree()), dtype=int)
    path_length, sampled_diameter = sampled_path_metrics(lcc, path_sources, seed)
    return ({'nodes': graph.number_of_nodes(), 'edges': graph.number_of_edges(), 'isolated_nodes': int(np.count_nonzero(degrees == 0)), 'largest_component_nodes': lcc.number_of_nodes(), 'mean_degree': float(degrees.mean()), 'average_local_clustering': float(nx.average_clustering(graph)), 'global_clustering': float(nx.transitivity(graph)), 'average_path_length_estimate': path_length, 'sampled_diameter_lower_bound': sampled_diameter, 'degree_assortativity': float(nx.degree_assortativity_coefficient(graph))}, lcc)

def rich_club_phi(graph: nx.Graph, thresholds: Iterable[int]) -> dict[int, float]:
    degrees = dict(graph.degree())
    values: dict[int, float] = {}
    for threshold in thresholds:
        rich = {node for node, degree in degrees.items() if degree > threshold}
        n_rich = len(rich)
        if n_rich < 2:
            continue
        edges = graph.subgraph(rich).number_of_edges()
        values[int(threshold)] = 2.0 * edges / (n_rich * (n_rich - 1))
    return values

def normalized_rich_club(graph: nx.Graph, null_models: int, swaps_per_edge: float, seed: int) -> list[dict[str, float]]:
    max_degree = max(dict(graph.degree()).values())
    thresholds = np.unique(np.geomspace(1, max_degree, num=30).astype(int))
    observed = rich_club_phi(graph, thresholds)
    if null_models == 0:
        return [{'threshold': threshold, 'phi': value, 'null_mean': float('nan'), 'normalized': float('nan')} for threshold, value in observed.items()]
    rng = np.random.default_rng(seed)
    null_values: dict[int, list[float]] = {threshold: [] for threshold in observed}
    source_edges = set(((min(u, v), max(u, v)) for u, v in graph.edges()))
    attempts = max(1, round(swaps_per_edge * graph.number_of_edges()))
    for _ in range(null_models):
        randomized_edges = set(source_edges)
        degree_preserving_rewire(randomized_edges, rng, attempts)
        randomized = nx.Graph()
        randomized.add_nodes_from(graph.nodes())
        randomized.add_edges_from(randomized_edges)
        phi = rich_club_phi(randomized, observed)
        for threshold in observed:
            null_values[threshold].append(phi.get(threshold, 0.0))
    result = []
    for threshold, value in observed.items():
        expected = float(np.mean(null_values[threshold]))
        result.append({'threshold': threshold, 'phi': value, 'null_mean': expected, 'normalized': value / expected if expected > 0 else float('nan')})
    return result

class UnionFind:

    def __init__(self, n_nodes: int) -> None:
        self.parent = list(range(n_nodes))
        self.size = [1] * n_nodes

    def find(self, node: int) -> int:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, first: int, second: int) -> int:
        root_first, root_second = (self.find(first), self.find(second))
        if root_first == root_second:
            return self.size[root_first]
        if self.size[root_first] < self.size[root_second]:
            root_first, root_second = (root_second, root_first)
        self.parent[root_second] = root_first
        self.size[root_first] += self.size[root_second]
        return self.size[root_first]

def static_attack_curve(graph: nx.Graph, removal_order: list[int], points: int=101) -> list[dict[str, float]]:
    n_nodes = graph.number_of_nodes()
    adjacency = {node: list(graph.neighbors(node)) for node in graph}
    active = np.zeros(n_nodes, dtype=bool)
    union_find = UnionFind(n_nodes)
    largest = 0
    curve = [(1.0, 0.0)]
    for surviving, node in enumerate(reversed(removal_order), start=1):
        active[node] = True
        largest = max(largest, 1)
        for neighbor in adjacency[node]:
            if active[neighbor]:
                largest = max(largest, union_find.union(node, neighbor))
        curve.append((1.0 - surviving / n_nodes, largest / n_nodes))
    indices = np.unique(np.linspace(0, len(curve) - 1, points).astype(int))
    return [{'fraction_removed': curve[index][0], 'fraction_surviving': 1.0 - curve[index][0], 'largest_component_fraction': curve[index][1]} for index in indices[::-1]]

def attack_orders(graph: nx.Graph, lcc: nx.Graph, betweenness_samples: int, seed: int) -> dict[str, list[int]]:
    nodes = list(graph.nodes())
    rng = np.random.default_rng(seed)
    degree = dict(graph.degree())
    core = nx.core_number(graph)
    betweenness_lcc = nx.betweenness_centrality(lcc, k=min(betweenness_samples, lcc.number_of_nodes()), normalized=False, seed=seed)
    betweenness = {node: betweenness_lcc.get(node, 0.0) for node in nodes}
    return {'random': list(rng.permutation(nodes)), 'degree': sorted(nodes, key=lambda node: (degree[node], node), reverse=True), 'betweenness': sorted(nodes, key=lambda node: (betweenness[node], node), reverse=True), 'k_coreness': sorted(nodes, key=lambda node: (core[node], node), reverse=True)}

def motter_lai_cascade(graph: nx.Graph, alpha: float, load_sources: int, seed: int, max_generations: int, prepared: tuple[nx.Graph, int, list[int], dict[int, float]] | None=None) -> dict[str, float]:
    if prepared is None:
        prepared = prepare_motter_lai_cascade(graph, load_sources, seed)
    initial, trigger, source_cohort, initial_load = prepared
    capacities = {node: (1.0 + alpha) * load for node, load in initial_load.items()}
    current = initial.copy()
    current.remove_node(trigger)
    failed = {trigger}
    generations = 0
    converged = False
    while current.number_of_nodes() > 0 and generations < max_generations:
        generations += 1
        load = fixed_cohort_load(current, source_cohort)
        overloaded = [node for node, value in load.items() if value > capacities[node]]
        if not overloaded:
            converged = True
            break
        current.remove_nodes_from(overloaded)
        failed.update(overloaded)
    final_lcc = max((len(component) for component in nx.connected_components(current)), default=0)
    return {'alpha': alpha, 'failed_nodes': len(failed), 'failed_fraction': len(failed) / initial.number_of_nodes(), 'largest_component_fraction': final_lcc / initial.number_of_nodes(), 'generations': generations, 'converged': converged}

def fixed_cohort_load(graph: nx.Graph, source_cohort: list[int]) -> dict[int, float]:
    active_sources = [node for node in source_cohort if node in graph]
    if not active_sources:
        return dict.fromkeys(graph, 0.0)
    estimate = nx.betweenness_centrality_subset(graph, sources=active_sources, targets=list(graph), normalized=False)
    scale = len(source_cohort) / len(active_sources)
    return {node: scale * value for node, value in estimate.items()}

def prepare_motter_lai_cascade(graph: nx.Graph, load_sources: int, seed: int) -> tuple[nx.Graph, int, list[int], dict[int, float]]:
    initial = largest_component(graph)
    trigger = max(initial.degree(), key=lambda pair: pair[1])[0]
    rng = np.random.default_rng(seed)
    candidate_sources = [node for node in initial if node != trigger]
    source_cohort = [int(node) for node in rng.choice(candidate_sources, size=min(load_sources, len(candidate_sources)), replace=False)]
    return (initial, trigger, source_cohort, fixed_cohort_load(initial, source_cohort))

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def plot_degree_distribution(graph: nx.Graph, path: Path) -> None:
    degree = np.fromiter((value for _, value in graph.degree()), dtype=int)
    positive_degree = degree[degree > 0]
    bin_edges = np.geomspace(1, positive_degree.max() + 1, num=25)
    counts, bin_edges = np.histogram(positive_degree, bins=bin_edges)
    bin_widths = np.diff(bin_edges)
    bin_centres = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    density = counts / (graph.number_of_nodes() * bin_widths)
    observed = counts > 0
    scaling_window = observed & (bin_centres >= 5) & (bin_centres <= 100)
    reference_scale = np.median(density[scaling_window] * bin_centres[scaling_window])
    reference_degree = np.geomspace(bin_centres[observed].min(), bin_centres[observed].max(), num=200)
    fig, axis = plt.subplots(figsize=(3.1, 2.25))
    axis.plot(np.log(bin_centres[observed]), np.log(density[observed]), 'o-', markersize=3.2, linewidth=1.0, color='#870011', label='synthetic model')
    axis.plot(np.log(reference_degree), np.log(reference_scale * reference_degree ** (-1)), '--', color='#4a4a4a', linewidth=1.1, label='slope $-1$ reference')
    axis.set(xlabel='$\\ln k$', ylabel='$\\ln P(k)$')
    axis.set_title('Degree distribution $P(k)$', fontsize=9)
    axis.legend(frameon=False, fontsize=7)
    axis.grid(alpha=0.18, linewidth=0.5, which='major')
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def plot_rich_club(rows: list[dict[str, float]], path: Path) -> None:
    threshold = [row['threshold'] for row in rows]
    normalized = [row['normalized'] for row in rows]
    fig, axis = plt.subplots(figsize=(3.1, 2.25))
    axis.semilogx(threshold, normalized, 'o-', markersize=2.8, color='#870011')
    axis.axhline(1.0, color='black', linewidth=0.8, linestyle='--')
    axis.set(xlabel='degree threshold $k$', ylabel='$\\phi(k)/\\tilde{\\phi}(k)$')
    axis.set_title('Rich-club ratio vs degree $k$', fontsize=9)
    axis.grid(alpha=0.2, which='both')
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def plot_attacks(curves: dict[str, list[dict[str, float]]], path: Path) -> None:
    styles = {'random': '#4c78a8', 'degree': '#f58518', 'betweenness': '#54a24b', 'k_coreness': '#e45756'}
    fig, axis = plt.subplots(figsize=(3.2, 2.25))
    for name, rows in curves.items():
        indices = np.unique(np.linspace(0, len(rows) - 1, min(31, len(rows))).astype(int))
        displayed_rows = [rows[index] for index in indices]
        axis.plot([row['fraction_surviving'] for row in displayed_rows], [row['largest_component_fraction'] for row in displayed_rows], label=name.replace('_', ' '), color=styles[name], marker='o', linestyle='None', markersize=3.2)
    axis.set(xlabel='$1-p_{\\mathrm{fail}}$', ylabel='relative LCC size', xlim=(0, 1), ylim=(0, 1))
    axis.set_title('Static failures', fontsize=9)
    axis.legend(frameon=False, fontsize=6.2, ncol=2)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('edge_list', type=Path)
    parser.add_argument('--n-nodes', type=int, default=5535)
    parser.add_argument('--output-dir', type=Path, default=Path('results/analysis_seed42'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--path-sources', type=int, default=256)
    parser.add_argument('--null-models', type=int, default=5)
    parser.add_argument('--swaps-per-edge', type=float, default=0.1)
    parser.add_argument('--betweenness-samples', type=int, default=64)
    return parser.parse_args()

def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    graph = load_graph(args.edge_list, args.n_nodes)
    metrics, lcc = structural_metrics(graph, args.path_sources, args.seed)
    rich_club = normalized_rich_club(graph, args.null_models, args.swaps_per_edge, args.seed)
    orders = attack_orders(graph, lcc, args.betweenness_samples, args.seed)
    attack_curves = {name: static_attack_curve(graph, order) for name, order in orders.items()}
    with (args.output_dir / 'metrics.json').open('w', encoding='utf-8') as file:
        json.dump({'synthetic': metrics, 'empirical_2015_reference': EMPIRICAL_2015, 'parameters': vars(args)}, file, indent=2, default=str)
    write_csv(args.output_dir / 'rich_club.csv', rich_club)
    for name, rows in attack_curves.items():
        write_csv(args.output_dir / f'attack_{name}.csv', rows)
    plot_degree_distribution(graph, args.output_dir / 'degree_distribution.png')
    plot_rich_club(rich_club, args.output_dir / 'rich_club.png')
    plot_attacks(attack_curves, args.output_dir / 'static_failures.png')

    print(json.dumps(metrics, indent=2))
if __name__ == '__main__':
    main()
