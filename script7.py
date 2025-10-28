import subprocess, os, re, math, random, csv
from datetime import datetime

# --- gem5 paths ---
GEM5_PATH = "./build/ARM/gem5.opt"
A76_SCRIPT_PATH = "resources_uarch_sim_assignment/scripts/CortexA76_scripts_gem5/CortexA76.py"
OUTPUTS_PATH = "results/annealing_mp3dec/"
os.makedirs(OUTPUTS_PATH, exist_ok=True)

# --- workload ---
WORKLOADS_BASE = "resources_uarch_sim_assignment/workloads/mp3_dec/"
BINARY = "mp3_dec"
WORKLOAD_PARAMS = "-w mp3dec_outfile.wav resources_uarch_sim_assignment/workloads/mp3_dec/mp3dec_testfile.mp3"

# --- stat to minimize ---
TARGET_STAT = r"simSeconds\s+(\d+(\.\d+)?)"

# --- search space ---
FU_BOUNDS = {
    "num_fu_FP_SIMD_ALU": (2, 8),
    "num_fu_intALU": (2, 8),
    "num_fu_write": (2, 6),
}

SQ_ENTRIES = [72, 144, 288]
L1D_SIZES = [32, 64, 128]  # kB

CSV_FILE = "annealing_results.csv"
CSV_EXPLORED = "explored_configs.csv"


# -------- helper functions --------
def cfg_key(cfg):
    return (
        cfg["num_fu_FP_SIMD_ALU"],
        cfg["num_fu_intALU"],
        cfg["num_fu_write"],
        cfg["sq_entries"],
        cfg["l1d_size"],
    )

def outdir_for(cfg):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return os.path.join(
        OUTPUTS_PATH,
        f"cfg_FPSIMD{cfg['num_fu_FP_SIMD_ALU']}_INT{cfg['num_fu_intALU']}_WR{cfg['num_fu_write']}"
        f"_SQ{cfg['sq_entries']}_L1D{cfg['l1d_size']}kB_{stamp}"
    )

def build_cmd(cfg, outdir):
    l2d_size = cfg["l1d_size"] * 2  # según tu bash original
    return (
        f"{GEM5_PATH} --outdir={outdir} {A76_SCRIPT_PATH} "
        f"--cmd={WORKLOADS_BASE}{BINARY} "
        f'--options="{WORKLOAD_PARAMS}" '
        f"--num_fu_FP_SIMD_ALU={cfg['num_fu_FP_SIMD_ALU']} "
        f"--num_fu_intALU={cfg['num_fu_intALU']} "
        f"--num_fu_write={cfg['num_fu_write']} "
        f"--sq_entries={cfg['sq_entries']} "
        f"--l1d_size={cfg['l1d_size']}kB "
        f"--l2d_size={l2d_size}kB "
        f"--l1i_size=32kB"
    )

def parse_sim_seconds(stats_path):
    if not os.path.exists(stats_path):
        return None
    with open(stats_path, "r") as f:
        for line in f:
            m = re.search(TARGET_STAT, line)
            if m:
                return float(m.group(1))
    return None

def run_config(cfg, cache):
    key = cfg_key(cfg)
    if key in cache:
        print("Cache hit:", cfg, "cost:", cache[key])
        return cache[key], None

    outdir = outdir_for(cfg)
    cmd = build_cmd(cfg, outdir)
    print("Launching:", cmd)
    p = subprocess.Popen(cmd, shell=True)
    p.wait()
    stats_path = os.path.join(outdir, "stats.txt")
    cost = parse_sim_seconds(stats_path)
    if cost is None or cost == 0.0:
        cost = float("inf")
    print("Finished config:", cfg, "cost:", cost)
    cache[key] = cost
    return cost, outdir

def all_neighbors(cfg, max_neighbors=10):
    neighs = []
    for k, (lo, hi) in FU_BOUNDS.items():
        for step in [-1, 1]:
            v = cfg[k] + step
            if lo <= v <= hi:
                new_cfg = dict(cfg)
                new_cfg[k] = v
                neighs.append(new_cfg)

    sq_idx = SQ_ENTRIES.index(cfg["sq_entries"])
    for step in [-1, 1]:
        i = sq_idx + step
        if 0 <= i < len(SQ_ENTRIES):
            new_cfg = dict(cfg)
            new_cfg["sq_entries"] = SQ_ENTRIES[i]
            neighs.append(new_cfg)

    l1d_idx = L1D_SIZES.index(cfg["l1d_size"])
    for step in [-1, 1]:
        i = l1d_idx + step
        if 0 <= i < len(L1D_SIZES):
            new_cfg = dict(cfg)
            new_cfg["l1d_size"] = L1D_SIZES[i]
            neighs.append(new_cfg)

    while len(neighs) < max_neighbors:
        new_cfg = dict(cfg)
        for k, (lo, hi) in FU_BOUNDS.items():
            new_cfg[k] = random.randint(lo, hi)
        new_cfg["sq_entries"] = random.choice(SQ_ENTRIES)
        new_cfg["l1d_size"] = random.choice(L1D_SIZES)
        neighs.append(new_cfg)

    random.shuffle(neighs)
    return neighs[:max_neighbors]

def run_parallel(candidates, cache):
    procs = {}
    results = {}

    for cfg in candidates:
        key = cfg_key(cfg)
        if key in cache:
            results[key] = (cfg, cache[key], None)
            continue
        outdir = outdir_for(cfg)
        cmd = build_cmd(cfg, outdir)
        print("Launching neighbor:", cfg)
        p = subprocess.Popen(cmd, shell=True)
        procs[p] = (cfg, outdir)

    for p, (cfg, outdir) in procs.items():
        p.wait()
        stats_path = os.path.join(outdir, "stats.txt")
        cost = parse_sim_seconds(stats_path)
        if cost is None or cost == 0.0:
            cost = float("inf")
        cache[cfg_key(cfg)] = cost
        results[cfg_key(cfg)] = (cfg, cost, outdir)
        print("Neighbor finished:", cfg, "cost:", cost)

    return results


# -------- annealing loop --------
def anneal_all_neighbors(start_cfg, rounds=15, T0=10.0, alpha=0.9, max_neighbors=10):
    cache = {}
    curr_cfg = dict(start_cfg)
    curr_cost, _ = run_config(curr_cfg, cache)
    best_cfg, best_cost = dict(curr_cfg), curr_cost
    T = T0

    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "best_cfg", "best_cost"])

        for r in range(rounds):
            neighbors = all_neighbors(curr_cfg, max_neighbors=max_neighbors)
            results = run_parallel(neighbors, cache)

            print(f"\n=== Round {r} ===")
            accepted = []
            for _, (cfg, cost, _) in results.items():
                delta = cost - curr_cost
                prob = math.exp(-delta / max(T, 1e-9)) if delta > 0 else 1.0
                if delta < 0 or prob > random.random():
                    accepted.append((cfg, cost))

            if accepted:
                chosen, chosen_cost = random.choice(accepted)
            else:
                chosen, chosen_cost = curr_cfg, curr_cost

            curr_cfg, curr_cost = chosen, chosen_cost
            if curr_cost < best_cost:
                best_cfg, best_cost = dict(curr_cfg), curr_cost

            writer.writerow([r, best_cfg, best_cost])
            f.flush()
            print(f"End of round {r}: curr_cost={curr_cost:.4f}, best={best_cost:.4f}, T={T:.3f}")
            T *= alpha

    print("\nFinished. Best config:", best_cfg, "cost:", best_cost)
    with open(CSV_EXPLORED, "w", newline="") as f2:
        writer2 = csv.writer(f2)
        writer2.writerow(["FP_SIMD_ALU", "INT_ALU", "WRITE", "SQ", "L1D", "cost"])
        for (f, i, w, sq, l1d), cost in cache.items():
            writer2.writerow([f, i, w, sq, l1d, cost])

    return best_cfg, best_cost


# -------- main --------
if __name__ == "__main__":
    random.seed(0)
    start = {
        "num_fu_FP_SIMD_ALU": 2,
        "num_fu_intALU": 2,
        "num_fu_write": 1,
        "sq_entries": 72,
        "l1d_size": 64,
    }
    best_cfg, best_cost = anneal_all_neighbors(start, rounds=12, T0=8.0, alpha=0.92, max_neighbors=10)
    print("\nBest configuration:", best_cfg, "cost:", best_cost)
