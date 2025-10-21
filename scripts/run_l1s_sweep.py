import subprocess
import os

GEM5_BIN = os.path.expanduser("~/gem5/build/ARM/gem5.fast")
SCRIPT_PATH = os.path.expanduser("~/gem5/resources_uarch_sim_assignment/scripts/CortexA76_scripts_gem5/CortexA76.py")
WORKLOAD_DIR = os.path.expanduser("~/gem5/resources_uarch_sim_assignment/workloads/jpeg2k_dec")

OUTPUT_ROOT = os.path.join(".", "results_l1d_sweep")

# ========= Configuración del workload ==========
binary = "jpeg2k_dec"
options = "-i jpg2dec_testfile.j2k -o jpg2dec_outfile.bmp"

# ========== Parametros ==========
l1d_sizes = ["32kB", "64kB", "128kB"]
fixed_opts = ["--l1i_size=32kB"]   # Como en el script simulate.sh

os.makedirs(OUTPUT_ROOT, exist_ok=True)

for size in l1d_sizes:
    folder_name = f"l1d_{size}"

    output_dir = os.path.join(OUTPUT_ROOT, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Ejecutando simulación con L1D size: {size}")

    cmd = [
        GEM5_BIN,
        "--outdir", output_dir, 
        SCRIPT_PATH, 
        "--cmd", binary, 
        "--options", options,
    ] + fixed_opts + [f"--l1d_size={size}"]

    # Ejecutar la simulación
    subprocess.run(cmd, cwd=WORKLOAD_DIR)
    