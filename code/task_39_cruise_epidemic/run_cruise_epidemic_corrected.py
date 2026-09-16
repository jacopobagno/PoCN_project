from __future__ import annotations
import argparse
import csv
import json
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
S, E, I, R = (0, 1, 2, 3)
TEMPORAL_REPRESENTATIONS = {'empirical_temporal', 'temporal_er'}
_WORKER_NETWORKS: dict[str, 'NetworkData'] | None = None
COLOURS = {'empirical_temporal': '#2f4858', 'aggregated_static': '#d95f02', 'temporal_er': '#7570b3', 'static_er': '#1b9e77'}
LABELS = {'empirical_temporal': 'empirical temporal', 'aggregated_static': 'aggregated static', 'temporal_er': 'temporal ER', 'static_er': 'static ER'}

@dataclass(frozen=True)
class NetworkData:
    name: str
    n_nodes: int
    temporal_edges: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]
    static_edges: tuple[np.ndarray, np.ndarray, np.ndarray]
    days: list[int]

def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nodes', type=Path, required=True, help='Processed nodes_sail_<sail>.csv file.')
    parser.add_argument('--temporal-edges', type=Path, required=True, help='Processed temporal_edges_sail_<sail>.csv file.')
    parser.add_argument('--aggregated-edges', type=Path, required=True, help='Processed aggregated_edges_sail_<sail>.csv file.')
    parser.add_argument('--output-dir', type=Path, default=Path('results/cruise_epidemic_corrected'))
    parser.add_argument('--runs', type=int, default=1000)
    parser.add_argument('--seed-runs', type=int, default=0, help='Runs per beta, representation, and seed strategy for the corrected seed-node experiment.')
    parser.add_argument('--seed-effects-only', action='store_true', help='Run only the corrected seed-node experiment; requires --seed-runs > 0.')
    parser.add_argument('--workers', type=int, default=1, help='Parallel worker processes for independent Monte Carlo batches (default: 1).')
    parser.add_argument('--horizon', type=int, default=21)
    parser.add_argument('--betas', default='0.02,0.05,0.08,0.12,0.16,0.22,0.30,0.40')
    parser.add_argument('--sigma', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=1.0 / 3.0)
    parser.add_argument('--duration-scale', type=float, default=1800.0)
    parser.add_argument('--seed', type=int, default=7)
    return parser.parse_args()

def read_empirical_network(nodes_path: Path, temporal_path: Path, aggregated_path: Path) -> NetworkData:
    nodes = pd.read_csv(nodes_path, dtype={'nodeID': str})
    temporal = pd.read_csv(temporal_path, dtype={'nodeID_from': str, 'nodeID_to': str})
    aggregated = pd.read_csv(aggregated_path, dtype={'nodeID_from': str, 'nodeID_to': str})
    node_ids = nodes['nodeID'].astype(str).to_numpy()
    index = {node: position for position, node in enumerate(node_ids)}
    days = sorted((int(day) for day in temporal['time'].unique()))
    temporal_edges: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for day, frame in temporal.groupby('time', sort=True):
        u = frame['nodeID_from'].map(index).to_numpy(dtype=int)
        v = frame['nodeID_to'].map(index).to_numpy(dtype=int)
        w = frame['contact_duration'].to_numpy(dtype=float)
        temporal_edges[int(day)] = (u, v, w)
    u = aggregated['nodeID_from'].map(index).to_numpy(dtype=int)
    v = aggregated['nodeID_to'].map(index).to_numpy(dtype=int)
    mean_daily_weight = (aggregated['weight'] / len(days)).to_numpy(dtype=float)
    return NetworkData('empirical', len(node_ids), temporal_edges, (u, v, mean_daily_weight), days)

def random_simple_pairs(n_nodes: int, edges: int, rng: np.random.Generator) -> np.ndarray:
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < edges:
        left = int(rng.integers(0, n_nodes))
        right = int(rng.integers(0, n_nodes - 1))
        if right >= left:
            right += 1
        if left > right:
            left, right = (right, left)
        pairs.add((left, right))
    return np.asarray(list(pairs), dtype=int)

def make_temporal_er(base: NetworkData, rng: np.random.Generator) -> NetworkData:
    temporal_edges: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for day, (_, _, weights) in base.temporal_edges.items():
        pairs = random_simple_pairs(base.n_nodes, len(weights), rng)
        temporal_edges[day] = (pairs[:, 0], pairs[:, 1], rng.permutation(weights))
    empty = (np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0, dtype=float))
    return NetworkData('temporal_er', base.n_nodes, temporal_edges, empty, base.days)

def make_static_er(base: NetworkData, rng: np.random.Generator) -> NetworkData:
    _, _, weights = base.static_edges
    pairs = random_simple_pairs(base.n_nodes, len(weights), rng)
    static_edges = (pairs[:, 0], pairs[:, 1], rng.permutation(weights))
    return NetworkData('static_er', base.n_nodes, {}, static_edges, base.days)

def transmit(states: np.ndarray, u: np.ndarray, v: np.ndarray, weights: np.ndarray, beta: float, duration_scale: float, rng: np.random.Generator) -> None:
    infectious_u = states[u] == I
    infectious_v = states[v] == I
    susceptible_u = states[u] == S
    susceptible_v = states[v] == S
    targets = np.concatenate([u[infectious_v & susceptible_u], v[infectious_u & susceptible_v]])
    if len(targets) == 0:
        return
    exposure_weights = np.concatenate([weights[infectious_v & susceptible_u], weights[infectious_u & susceptible_v]])
    probability = 1.0 - np.exp(-beta * exposure_weights / duration_scale)
    states[targets[rng.random(len(targets)) < probability]] = E

def run_one(network: NetworkData, representation: str, beta: float, horizon: int, sigma: float, gamma: float, duration_scale: float, rng: np.random.Generator, seed_pool: np.ndarray | None=None) -> float:
    states = np.full(network.n_nodes, S, dtype=np.int8)
    if seed_pool is None:
        seed_node = int(rng.integers(0, network.n_nodes))
    else:
        seed_node = int(rng.choice(seed_pool))
    states[seed_node] = I
    for step in range(horizon):
        if representation in TEMPORAL_REPRESENTATIONS:
            day = network.days[step % len(network.days)]
            u, v, weights = network.temporal_edges[day]
        else:
            u, v, weights = network.static_edges
        transmit(states, u, v, weights, beta, duration_scale, rng)
        exposed = np.flatnonzero(states == E)
        states[exposed[rng.random(len(exposed)) < sigma]] = I
        infectious = np.flatnonzero(states == I)
        states[infectious[rng.random(len(infectious)) < gamma]] = R
        if not np.any((states == E) | (states == I)):
            break
    return float(np.mean(states != S))

def initialise_worker(base: NetworkData, surrogate_seed: int) -> None:
    global _WORKER_NETWORKS
    surrogate_rng = np.random.default_rng(surrogate_seed)
    _WORKER_NETWORKS = {'empirical_temporal': base, 'aggregated_static': base, 'temporal_er': make_temporal_er(base, surrogate_rng), 'static_er': make_static_er(base, surrogate_rng)}

def simulate_batch(task: tuple[str, float, int, int, float, float, float, int, int]) -> list[dict[str, float | int | str]]:
    representation, beta, count, horizon, sigma, gamma, duration_scale, seed, first_run = task
    if _WORKER_NETWORKS is None:
        raise RuntimeError('Worker networks were not initialised.')
    network = _WORKER_NETWORKS[representation]
    rng = np.random.default_rng(seed)
    return [{'representation': representation, 'beta': beta, 'run': first_run + run, 'attack_rate': run_one(network, representation, beta, horizon, sigma, gamma, duration_scale, rng)} for run in range(count)]

def simulate(base: NetworkData, betas: list[float], runs: int, horizon: int, sigma: float, gamma: float, duration_scale: float, seed: int, workers: int) -> list[dict[str, float | int | str]]:
    if workers < 1:
        raise ValueError('--workers must be positive.')
    rng = np.random.default_rng(seed)
    temporal_er = make_temporal_er(base, rng)
    static_er = make_static_er(base, rng)
    networks = {'empirical_temporal': base, 'aggregated_static': base, 'temporal_er': temporal_er, 'static_er': static_er}
    if workers == 1:
        rows: list[dict[str, float | int | str]] = []
        for beta in betas:
            for representation, network in networks.items():
                for run in range(runs):
                    attack_rate = run_one(network, representation, beta, horizon, sigma, gamma, duration_scale, rng)
                    rows.append({'representation': representation, 'beta': beta, 'run': run, 'attack_rate': attack_rate})
        return rows
    batch_size = max(1, (runs + workers - 1) // workers)
    task_specs: list[tuple[str, float, int, int, float, float, float, int, int]] = []
    child_sequences = np.random.SeedSequence(seed).spawn(len(betas) * len(networks) * ((runs + batch_size - 1) // batch_size))
    sequence_index = 0
    for beta in betas:
        for representation in networks:
            for first_run in range(0, runs, batch_size):
                child_seed = int(child_sequences[sequence_index].generate_state(1, dtype=np.uint64)[0])
                sequence_index += 1
                task_specs.append((representation, beta, min(batch_size, runs - first_run), horizon, sigma, gamma, duration_scale, child_seed, first_run))
    surrogate_seed = int(np.random.SeedSequence(seed).spawn(1)[0].generate_state(1, dtype=np.uint64)[0])
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=initialise_worker, initargs=(base, surrogate_seed)) as executor:
        for batch in executor.map(simulate_batch, task_specs):
            rows.extend(batch)
    return rows

def seed_pools(nodes_path: Path, base: NetworkData) -> dict[str, np.ndarray]:
    nodes = pd.read_csv(nodes_path, dtype={'nodeID': str})
    if len(nodes) != base.n_nodes:
        raise ValueError('The node file does not match the empirical network.')
    roles = nodes['group_or_role'].astype(str).to_numpy()
    u, v, _ = base.static_edges
    degrees = np.bincount(np.concatenate((u, v)), minlength=base.n_nodes)
    positive_degrees = degrees[degrees > 0]
    if len(positive_degrees) == 0:
        high_degree = np.arange(base.n_nodes)
        low_degree = np.arange(base.n_nodes)
    else:
        high_threshold = np.quantile(positive_degrees, 0.9)
        low_threshold = np.quantile(positive_degrees, 0.1)
        high_degree = np.flatnonzero(degrees >= high_threshold)
        low_degree = np.flatnonzero((degrees > 0) & (degrees <= low_threshold))
    pools = {'random': np.arange(base.n_nodes), 'high_degree_top_10pct': high_degree, 'low_degree_bottom_10pct': low_degree, 'passenger': np.flatnonzero(roles == 'P'), 'crew': np.flatnonzero(roles == 'C')}
    return {name: pool for name, pool in pools.items() if len(pool) > 0}

def initialise_seed_worker(base: NetworkData) -> None:
    global _WORKER_NETWORKS
    _WORKER_NETWORKS = {'empirical_temporal': base, 'aggregated_static': base}

def simulate_seed_batch(task: tuple[str, str, np.ndarray, float, int, int, float, float, float, int, int]) -> list[dict[str, float | int | str]]:
    representation, strategy, pool, beta, count, horizon, sigma, gamma, duration_scale, seed, first_run = task
    if _WORKER_NETWORKS is None:
        raise RuntimeError('Worker networks were not initialised.')
    network = _WORKER_NETWORKS[representation]
    rng = np.random.default_rng(seed)
    return [{'representation': representation, 'seed_strategy': strategy, 'seed_pool_size': len(pool), 'beta': beta, 'run': first_run + run, 'attack_rate': run_one(network, representation, beta, horizon, sigma, gamma, duration_scale, rng, seed_pool=pool)} for run in range(count)]

def simulate_seed_effects(base: NetworkData, nodes_path: Path, betas: list[float], runs: int, horizon: int, sigma: float, gamma: float, duration_scale: float, seed: int, workers: int) -> list[dict[str, float | int | str]]:
    if runs < 1 or workers < 1:
        raise ValueError('Seed runs and workers must be positive.')
    pools = seed_pools(nodes_path, base)
    representations = ('empirical_temporal', 'aggregated_static')
    if workers == 1:
        rng = np.random.default_rng(seed)
        rows: list[dict[str, float | int | str]] = []
        for beta in betas:
            for representation in representations:
                for strategy, pool in pools.items():
                    for run in range(runs):
                        rows.append({'representation': representation, 'seed_strategy': strategy, 'seed_pool_size': len(pool), 'beta': beta, 'run': run, 'attack_rate': run_one(base, representation, beta, horizon, sigma, gamma, duration_scale, rng, seed_pool=pool)})
        return rows
    batch_size = max(1, (runs + workers - 1) // workers)
    batches_per_condition = (runs + batch_size - 1) // batch_size
    task_specs: list[tuple[str, str, np.ndarray, float, int, int, float, float, float, int, int]] = []
    child_sequences = np.random.SeedSequence(seed).spawn(len(betas) * len(representations) * len(pools) * batches_per_condition)
    sequence_index = 0
    for beta in betas:
        for representation in representations:
            for strategy, pool in pools.items():
                for first_run in range(0, runs, batch_size):
                    child_seed = int(child_sequences[sequence_index].generate_state(1, dtype=np.uint64)[0])
                    sequence_index += 1
                    task_specs.append((representation, strategy, pool, beta, min(batch_size, runs - first_run), horizon, sigma, gamma, duration_scale, child_seed, first_run))
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=initialise_seed_worker, initargs=(base,)) as executor:
        for batch in executor.map(simulate_seed_batch, task_specs):
            rows.extend(batch)
    return rows

def summarize_seed_effects(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    frame = pd.DataFrame(rows)
    summary = frame.groupby(['representation', 'seed_strategy', 'seed_pool_size', 'beta'], as_index=False).agg(mean_attack_rate=('attack_rate', 'mean'), sd_attack_rate=('attack_rate', 'std'), runs=('attack_rate', 'size')).sort_values(['representation', 'seed_strategy', 'beta'])
    return summary.to_dict(orient='records')

def plot_seed_effects(summary: list[dict[str, float | int | str]], output_dir: Path) -> None:
    frame = pd.DataFrame(summary)
    strategy_labels = {'random': 'random', 'high_degree_top_10pct': 'high degree', 'low_degree_bottom_10pct': 'low degree', 'passenger': 'passenger', 'crew': 'crew'}
    strategy_colours = {'random': '#2f4858', 'high_degree_top_10pct': '#d95f02', 'low_degree_bottom_10pct': '#1b9e77', 'passenger': '#7570b3', 'crew': '#e7298a'}
    titles = {'empirical_temporal': 'Empirical temporal network', 'aggregated_static': 'Aggregated static network'}
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.75), sharey=True)
    for axis, representation in zip(axes, ('empirical_temporal', 'aggregated_static')):
        subset = frame[frame['representation'] == representation]
        for strategy, values in subset.groupby('seed_strategy'):
            axis.plot(values['beta'], values['mean_attack_rate'], 'o-', linewidth=1.4, markersize=3.5, color=strategy_colours[strategy], label=strategy_labels[strategy])
        axis.set(title=titles[representation], xlabel='transmission parameter $\\beta$', ylim=(0, 1.02))
        axis.grid(alpha=0.2)
    axes[0].set_ylabel('mean final attack rate')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=7.4, frameon=False)
    fig.tight_layout(rect=(0, 0.15, 1, 1), w_pad=2.0)
    fig.savefig(output_dir / 'corrected_seed_strategy_attack_rate.png', dpi=300)
    plt.close(fig)

def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    frame = pd.DataFrame(rows)
    summary = frame.groupby(['representation', 'beta'], as_index=False)['attack_rate'].agg(['mean', 'std', 'count']).reset_index().rename(columns={'mean': 'mean_attack_rate', 'std': 'sd_attack_rate', 'count': 'runs'}).sort_values(['representation', 'beta'])
    return summary.to_dict(orient='records')

def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def plot_results(summary: list[dict[str, float | int | str]], output_dir: Path) -> None:
    frame = pd.DataFrame(summary)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.35))
    for representation, group in frame.groupby('representation'):
        confidence_interval = 1.96 * group['sd_attack_rate'] / np.sqrt(group['runs'])
        axes[0].errorbar(group['beta'], group['mean_attack_rate'], yerr=confidence_interval, fmt='o-', linewidth=1.5, markersize=3.5, capsize=2.0, elinewidth=0.8, color=COLOURS[representation], label=LABELS[representation])
    axes[0].set_title('Attack rate vs transmission probability $\\beta$', fontsize=9)
    axes[0].set(xlabel='$\\beta$', ylabel='Mean attack rate', ylim=(0, 1.02))
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=6.6, loc='lower right')
    empirical = frame[frame['representation'] == 'empirical_temporal'][['beta', 'mean_attack_rate']].rename(columns={'mean_attack_rate': 'empirical'})
    gaps = frame.merge(empirical, on='beta', how='inner')
    for representation, group in gaps[gaps['representation'] != 'empirical_temporal'].groupby('representation'):
        axes[1].plot(group['beta'], group['mean_attack_rate'] - group['empirical'], 'o-', linewidth=1.5, markersize=3.5, color=COLOURS[representation], label=LABELS[representation])
    axes[1].axhline(0, color='#111111', linewidth=1.5, zorder=4, label='empirical network reference')
    axes[1].set_title('Relative difference from empirical network', fontsize=9)
    axes[1].set(xlabel='$\\beta$', ylabel='Relative difference')
    axes[1].set_ylim(bottom=-0.02)
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=6.6)
    fig.tight_layout(w_pad=2.2)
    fig.savefig(output_dir / 'corrected_temporal_comparison.png', dpi=300)
    plt.close(fig)



def main() -> None:
    args = arguments()
    if args.runs < 1 or args.seed_runs < 0 or args.horizon < 1 or (args.workers < 1):
        raise ValueError('--runs, --seed-runs, --horizon, and --workers must be valid positive values.')
    if args.seed_effects_only and args.seed_runs == 0:
        raise ValueError('--seed-effects-only requires --seed-runs > 0.')
    betas = [float(value) for value in args.betas.split(',') if value.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = read_empirical_network(args.nodes, args.temporal_edges, args.aggregated_edges)
    if not args.seed_effects_only:
        print(f'Running corrected comparison: {len(betas)} beta values, {args.runs} runs each, {args.workers} worker(s).', flush=True)
        rows = simulate(base, betas, args.runs, args.horizon, args.sigma, args.gamma, args.duration_scale, args.seed, args.workers)
        summary = summarize(rows)
        write_csv(args.output_dir / 'corrected_simulation_runs.csv', rows)
        write_csv(args.output_dir / 'corrected_simulation_summary.csv', summary)
        plot_results(summary, args.output_dir)
        metadata = {'correction': 'empirical_temporal is explicitly dispatched to day-specific temporal edges.', 'runs': args.runs, 'horizon': args.horizon, 'betas': betas, 'sigma': args.sigma, 'gamma': args.gamma, 'duration_scale_seconds': args.duration_scale, 'seed': args.seed, 'workers': args.workers, 'surrogates': 'One fixed temporal-ER and static-ER realization is shared across all runs and beta values.', 'nodes': base.n_nodes, 'observed_days': base.days}
        (args.output_dir / 'corrected_run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    if args.seed_runs > 0:
        print(f'Running corrected seed-node experiment: {len(betas)} beta values, {args.seed_runs} runs per condition, {args.workers} worker(s).', flush=True)
        seed_rows = simulate_seed_effects(base, args.nodes, betas, args.seed_runs, args.horizon, args.sigma, args.gamma, args.duration_scale, args.seed + 1, args.workers)
        seed_summary = summarize_seed_effects(seed_rows)
        write_csv(args.output_dir / 'corrected_seed_strategy_runs.csv', seed_rows)
        write_csv(args.output_dir / 'corrected_seed_strategy_summary.csv', seed_summary)
        plot_seed_effects(seed_summary, args.output_dir)
        seed_metadata = {'correction': 'empirical_temporal is explicitly dispatched to day-specific temporal edges.', 'protocol': 'Same five seed pools and parameters as the original seed-node extension.', 'runs_per_condition': args.seed_runs, 'representations': ['empirical_temporal', 'aggregated_static'], 'seed_strategies': {name: len(pool) for name, pool in seed_pools(args.nodes, base).items()}, 'horizon': args.horizon, 'betas': betas, 'sigma': args.sigma, 'gamma': args.gamma, 'duration_scale_seconds': args.duration_scale, 'seed': args.seed + 1, 'workers': args.workers, 'nodes': base.n_nodes, 'observed_days': base.days}
        (args.output_dir / 'corrected_seed_metadata.json').write_text(json.dumps(seed_metadata, indent=2), encoding='utf-8')
        print(f'Wrote corrected seed-node results to {args.output_dir}', flush=True)
    if not args.seed_effects_only:
        print(f'Wrote corrected results to {args.output_dir}', flush=True)
if __name__ == '__main__':
    main()
