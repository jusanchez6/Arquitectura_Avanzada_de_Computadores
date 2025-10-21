import itertools
import subprocess
import os

# paths 
GEM5_BIN = os.path.expanduser("~/gem5/build/ARM/gem5.fast")
SCRIPT_PATH = os.path.expanduser("~/gem5/resources_uarch_sim_assignment/scripts/CortexA76_scripts_gem5/CortexA76.py")
WORKLOAD_DIR = os.path.expanduser("~/gem5/resources_uarch_sim_assignment/workloads/jpeg2k_dec")

OUTPUT_ROOT = os.path.join(".", "results")

# Workload setup

binary = "jpeg2k_dec"
options = "-i jpg2dec_testfile.j2k -o jpg2dec_outfile.bmp"

# fixed options (from simulate.sh)
fixed_opts = ["--l1i_size=32kB", "--l1i_size=128kB"]


# ==================================================================
# DESIGN SPACE (DSE)
# ==================================================================


params = {
    "l1d_assoc": [4, 8, 16],                    # Grado de asociatividad de la memoria cache L1
    "l2_size":  ["256kB", "512kB", "1MB"],      # Tamaño de la cache L2
    "rob_entries": [128, 192, 256],             # entradas de la cola de reordenamiento (ROB)
}


# Run experiments

os.makedirs(OUTPUT_ROOT, exist_ok=True)
keys, values = zip(*params.items())

for combo in itertools.product(*values):
    config = dict(zip(keys, combo))
    folder_name = "_".join([f"{k}{v}" for k, v in config.items()])
    output_dir = os.path.join(OUTPUT_ROOT, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Running configuration: {folder_name}")

    cmd = [
            GEM5_BIN,
            "--outdir", output_dir, 
            SCRIPT_PATH, 
            "--cmd", binary, 
            "--options", options,

    ] + fixed_opts

    for k, v in config.items():
        cmd += [f"--{k}", str(v)]


    # run the simulation
    subprocess.run(cmd, cwd=WORKLOAD_DIR)




