from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Iterable
import numpy as np
DEFAULT_NODES = 5535
DEFAULT_EDGES = 274831

def canonical_edge(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)

def random_initial_edges(n_nodes: int, n_edges: int, rng: np.random.Generator) -> set[tuple[int, int]]:
    candidates = [(u, v) for u in range(n_nodes) for v in range(u + 1, n_nodes)]
    selected = rng.choice(len(candidates), size=n_edges, replace=False)
    return {candidates[index] for index in selected}

def edge_schedule(total_new_edges: int, n_steps: int) -> list[int]:
    base, remainder = divmod(total_new_edges, n_steps)
    return [base + int(step < remainder) for step in range(n_steps)]

def sample_endpoints(timestamps: np.ndarray, bandwidths: np.ndarray, time: int, n_endpoints: int, beta: float, gamma: float, rng: np.random.Generator) -> np.ndarray:
    ages = time - timestamps
    weights = np.power(ages, beta, dtype=float) * np.power(bandwidths, gamma)
    total_weight = weights.sum()
    if total_weight <= 0 or not np.isfinite(total_weight):
        raise ValueError('Invalid attachment weights; use beta >= 0 and gamma >= 0.')
    if n_endpoints > len(timestamps):
        raise ValueError('The seed graph is too small for the requested number of endpoints.')
    return rng.choice(len(timestamps), size=n_endpoints, replace=False, p=weights / total_weight)

def add_weighted_edges(edges: set[tuple[int, int]], timestamps: np.ndarray, bandwidths: np.ndarray, time: int, requested_edges: int, beta: float, gamma: float, rng: np.random.Generator) -> None:
    added = 0
    attempts = 0
    max_attempts = max(1000, 100 * requested_edges)
    while added < requested_edges:
        remaining = requested_edges - added
        endpoints = sample_endpoints(timestamps, bandwidths, time, 2 * remaining, beta, gamma, rng)
        rng.shuffle(endpoints)
        for left, right in endpoints.reshape(-1, 2):
            edge = canonical_edge(int(left), int(right))
            if edge not in edges:
                edges.add(edge)
                added += 1
        attempts += 1
        if attempts >= max_attempts:
            raise RuntimeError('Could not add enough distinct edges at this time step.')

def degree_preserving_rewire(edges: set[tuple[int, int]], rng: np.random.Generator, attempts: int) -> int:
    edge_list = list(edges)
    accepted = 0
    for _ in range(attempts):
        first, second = rng.choice(len(edge_list), size=2, replace=False)
        a, b = edge_list[int(first)]
        c, d = edge_list[int(second)]
        if rng.random() < 0.5:
            new_first, new_second = (canonical_edge(a, c), canonical_edge(b, d))
        else:
            new_first, new_second = (canonical_edge(a, d), canonical_edge(b, c))
        if new_first[0] == new_first[1] or new_second[0] == new_second[1]:
            continue
        if new_first == new_second:
            continue
        old_first, old_second = (edge_list[int(first)], edge_list[int(second)])
        edges.remove(old_first)
        edges.remove(old_second)
        if new_first in edges or new_second in edges:
            edges.add(old_first)
            edges.add(old_second)
            continue
        edges.add(new_first)
        edges.add(new_second)
        edge_list[int(first)] = new_first
        edge_list[int(second)] = new_second
        accepted += 1
    return accepted

def generate_darknet(n_nodes: int, n_edges: int, n0: int, e0: int, beta: float, gamma: float, bandwidth_mu: float, bandwidth_sigma: float, rewire_attempts: int, seed: int) -> tuple[set[tuple[int, int]], np.ndarray, np.ndarray, int]:
    if n0 < 2 or n0 >= n_nodes:
        raise ValueError('n0 must satisfy 2 <= n0 < n_nodes.')
    if e0 < 0 or e0 > n0 * (n0 - 1) // 2 or e0 >= n_edges:
        raise ValueError('e0 must be feasible and smaller than the target number of edges.')
    if beta < 0 or gamma < 0:
        raise ValueError('This implementation requires non-negative beta and gamma.')
    rng = np.random.default_rng(seed)
    timestamps = np.zeros(n_nodes, dtype=int)
    bandwidths = rng.lognormal(mean=bandwidth_mu, sigma=bandwidth_sigma, size=n_nodes)
    edges = random_initial_edges(n0, e0, rng)
    schedule = edge_schedule(n_edges - e0, n_nodes - n0)
    if 2 * max(schedule) > n0 + 1:
        raise ValueError('n0 is too small: the first growth step needs more distinct endpoints than the active graph contains. Increase n0 or e0.')
    for offset, n_new_edges in enumerate(schedule, start=1):
        node = n0 + offset - 1
        timestamps[node] = offset
        active = node + 1
        add_weighted_edges(edges, timestamps[:active], bandwidths[:active], offset, n_new_edges, beta, gamma, rng)
    if len(edges) != n_edges:
        raise RuntimeError(f'Expected {n_edges} edges, generated {len(edges)}.')
    accepted_swaps = degree_preserving_rewire(edges, rng, rewire_attempts)
    return (edges, timestamps, bandwidths, accepted_swaps)

def degrees_from_edges(n_nodes: int, edges: Iterable[tuple[int, int]]) -> np.ndarray:
    degrees = np.zeros(n_nodes, dtype=int)
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1
    return degrees

def write_outputs(output_prefix: Path, edges: set[tuple[int, int]], timestamps: np.ndarray, bandwidths: np.ndarray, config: dict) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with output_prefix.with_suffix('.edgelist').open('w', encoding='utf-8') as file:
        for u, v in sorted(edges):
            file.write(f'{u} {v}\n')
    degrees = degrees_from_edges(len(timestamps), edges)
    with output_prefix.with_suffix('.nodes.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['node', 'arrival_time', 'bandwidth', 'degree'])
        for node, (timestamp, bandwidth, degree) in enumerate(zip(timestamps, bandwidths, degrees)):
            writer.writerow([node, timestamp, bandwidth, degree])
    config['summary'] = {'nodes': int(len(timestamps)), 'edges': int(len(edges)), 'mean_degree': float(degrees.mean()), 'min_degree': int(degrees.min()), 'max_degree': int(degrees.max())}
    with output_prefix.with_suffix('.json').open('w', encoding='utf-8') as file:
        json.dump(config, file, indent=2)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-nodes', type=int, default=DEFAULT_NODES)
    parser.add_argument('--n-edges', type=int, default=DEFAULT_EDGES)
    parser.add_argument('--n0', type=int, default=200, help='Initial random-graph size.')
    parser.add_argument('--e0', type=int, default=1000, help='Initial random-graph edge count.')
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--bandwidth-mu', type=float, default=0.0)
    parser.add_argument('--bandwidth-sigma', type=float, default=1.0)
    parser.add_argument('--rewire-attempts', type=int, default=None, help='Double-edge-swap attempts; defaults to one attempt per edge.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-prefix', type=Path, default=Path('results/synthetic_darknet_2015_seed42'))
    return parser.parse_args()

def main() -> None:
    args = parse_arguments()
    rewire_attempts = args.n_edges if args.rewire_attempts is None else args.rewire_attempts
    edges, timestamps, bandwidths, accepted_swaps = generate_darknet(n_nodes=args.n_nodes, n_edges=args.n_edges, n0=args.n0, e0=args.e0, beta=args.beta, gamma=args.gamma, bandwidth_mu=args.bandwidth_mu, bandwidth_sigma=args.bandwidth_sigma, rewire_attempts=rewire_attempts, seed=args.seed)
    config = vars(args).copy()
    config['output_prefix'] = str(args.output_prefix)
    config['rewire_attempts'] = rewire_attempts
    config['accepted_swaps'] = accepted_swaps
    config['paper'] = 'De Domenico and Arenas, Phys. Rev. E 95, 022313 (2017)'
    write_outputs(args.output_prefix, edges, timestamps, bandwidths, config)
    print(json.dumps(config, indent=2))
if __name__ == '__main__':
    main()
