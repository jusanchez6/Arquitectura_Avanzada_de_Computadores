#%% 1. ============ Importar las librerias ==============================
import os 
import re
import matplotlib.pyplot as plt

#%% 2. ============ Carpetas y variables ==============================
WORKLOAD_DIR = os.path.expanduser("~/gem5/resources_uarch_sim_assignment/workloads/jpeg2k_dec")
RESULTS_DIR = os.path.join(".", "results")

# Metrica a medir
metric_name = "system.cpu.cpi"

sizes = []
cpi_values = []

#%% 3. ============ Obtener los resultados ==============================

for carpeta in sorted(os.listdir(RESULTS_DIR)):
    
    stats_path = os.path.join(RESULTS_DIR, carpeta, "stats.txt")
    if os.path.isfile(stats_path):
        continue

    with open(stats_path) as f:
        for line in f:
            if line.startswith(metric_name):
                value = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
                sizes.append(carpeta.split("_"[1]))
                cpi_values.append(value)
                break 



#%% 4. ============ Mostrar resultados ==============================
print("\nResultados de CPI por tamaño de L1D")

for t, c in zip(sizes, cpi_values):
    print(f"    L1D = {t:>6} -> CPI ={c:.4f}")

# ========== Grafica de los resultados ================
plt.figure(figsize=(6, 4))
plt.plot(sizes, cpi_values, marker='o', linestyle= '-')
plt.title("Efecto del tamaño de la cache L1D sobre el CPI.")
plt.xlabel("Tamaño de L1D")
plt.ylabel("CPI (ciclos por instrucción)")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "cpi_vs_l1d_size.svg"))
plt.show()
