#%% 1. Imortar librerias
import os
import numpy as np
import matplotlib.pyplot as plt

# Configurar tamaños de fuente y figuras más grandes
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

def find_pareto_optimal(cpi_values, energy_values):
    """Encuentra los puntos óptimos de Pareto (minimizar CPI y Energy)"""
    points = np.column_stack((cpi_values, energy_values))
    pareto_mask = np.ones(len(points), dtype=bool)
    
    for i, point in enumerate(points):
        # Un punto es dominado si existe otro que es mejor en ambos objetivos
        for j, other_point in enumerate(points):
            if i != j:
                # other_point domina a point si es menor o igual en ambos objetivos
                # y estrictamente menor en al menos uno
                if (other_point[0] <= point[0] and other_point[1] <= point[1] and 
                    (other_point[0] < point[0] or other_point[1] < point[1])):
                    pareto_mask[i] = False
                    break
    
    return pareto_mask

#%% 2. Definir rutas y métricas a extraer
results_dir = "results"
metrics = [
    "system.cpu.statFuBusy::SimdFloatAdd",
    "system.cpu.cpi",
    "system.cpu.ipc", 
    "system.cpu.dcache.demandMisses::total",
    "simSeconds"
]


# Arreglos para guardar los valores de cada métrica
folder_names = []
number_of_loads_values = []
cpi_values = []
ipc_values = []
demand_misses_values = []
simd_add_busy_values = []
sim_seconds_values = []

# Métricas de potencia
power_metrics = ["Peak Power", "Peak Dynamic", "Total Leakage", "Runtime Dynamic"]  # SIN espacios al inicio

peak_power_values = []
peak_dynamic_power_values = []
total_leakage_values = []
runtime_dynamic_values = []


#%% 3. Recolectar métricas específicas de cada carpeta
print("Recolectando métricas específicas...")
for folder_name in os.listdir(results_dir):
    folder_path = os.path.join(results_dir, folder_name)
    if os.path.isdir(folder_path):
        stats_file = os.path.join(folder_path, "stats.txt")
        
        if os.path.exists(stats_file):
            folder_names.append(folder_name)
            
            # Leer todo el archivo una vez
            with open(stats_file, 'r') as f:
                content = f.read().splitlines()
            
            # Buscar cada métrica en el contenido
            for metric in metrics:
                value_found = None
                for line in content:
                    if line.startswith(metric):
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                value_found = float(parts[1])
                                break
                            except ValueError:
                                value_found = None
                                break
                
                # Asignar el valor al arreglo correspondiente
                if value_found is not None:
                    if metric == "system.cpu.statFuBusy::SimdFloatAdd":
                        simd_add_busy_values.append(value_found)
                    elif metric == "system.cpu.cpi":
                        cpi_values.append(value_found)
                    elif metric == "system.cpu.ipc":
                        ipc_values.append(value_found)
                    elif metric == "system.cpu.dcache.demandMisses::total":
                        demand_misses_values.append(value_found)
                    elif metric == "simSeconds":
                        sim_seconds_values.append(value_found)
                    print(f"✅ {folder_name} - {metric}: {value_found}")
                else:
                    print(f"❌ {folder_name}: {metric} no encontrada")
        else:
            print(f"❌ {folder_name}: stats.txt no existe")

print(f"\n📊 Resumen:")
print(f"Carpetas procesadas: {len(folder_names)}")
print(f"Valores simd_add_busy: {len(simd_add_busy_values)}")
print(f"Valores cpi: {len(cpi_values)}")
print(f"Valores ipc: {len(ipc_values)}")
print(f"Valores demand_misses: {len(demand_misses_values)}")
print(f"Valores sim_seconds: {len(sim_seconds_values)}")


print("Recolectando metricas de potencia...")

for folder_name in os.listdir(results_dir):
    folder_path = os.path.join(results_dir, folder_name)
    if os.path.isdir(folder_path):
        power_file = os.path.join(folder_path, "power_results.txt")
        
        if os.path.exists(power_file):
            with open(power_file, 'r') as f:
                content = f.read().splitlines()
            
            for power_metric in power_metrics:
                value_found = None
                for line in content:
                    if power_metric in line:
                        parts = line.split("=")
                        if len(parts) >= 2:
                            try:
                                # Limpiar el valor (quitar " W" y espacios)
                                value_str = parts[1].strip().replace(" W", "")
                                value_found = float(value_str)
                                break
                            except ValueError:
                                value_found = None
                                break
                
                if value_found is not None:
                    if power_metric == "Peak Power":
                        peak_power_values.append(value_found)
                    elif power_metric == "Peak Dynamic":
                        peak_dynamic_power_values.append(value_found)
                    elif power_metric == "Total Leakage":
                        total_leakage_values.append(value_found)
                    elif power_metric == "Runtime Dynamic":
                        runtime_dynamic_values.append(value_found)
                    
                    print(f"✅ {folder_name} - {power_metric}: {value_found}")
                else:
                    print(f"❌ {folder_name}: {power_metric} no encontrada")
        else:
            print(f"❌ {folder_name}: power_results.txt no existe")        

print(f"\n📊 Resumen metricas de potencia:")
print(f"Carpetas procesadas: {len(folder_names)}")
print(f"Valores peak_power: {len(peak_power_values)}")
print(f"Valores peak_dynamic_power: {len(peak_dynamic_power_values)}")


# calcular el EDP de cada configuración

edp_values = []
energy_values = []
for i in range(len(folder_names)):
    if i < len(cpi_values) and i < len(peak_dynamic_power_values):
        energy = (total_leakage_values[i] + runtime_dynamic_values[i])  * cpi_values[i]  # Energía total
        energy_values.append(energy)
        edp = energy * cpi_values[i]  # EDP = Energía * CPI
        edp_values.append(edp)
    else:
        edp_values.append(None)

print(f"\n📊 Resumen EDP:"
      f"\nCarpetas procesadas: {len(folder_names)}"
      f"\nValores EDP: {len(edp_values)}"
      f"\nValores Energía: {len(energy_values)}"
      )

for i in range(len(folder_names)):
    if edp_values[i] is not None:
        print(f"✅ {folder_names[i]} - EDP: {edp_values[i]}")
    else:
        print(f"❌ {folder_names[i]} - EDP no calculado")

performance_values = [1/seconds if seconds != 0 else 0 for seconds in sim_seconds_values]

print(f"\n📊 Resumen Performance:")
print(f"Valores performance: {len(performance_values)}")


#%% graficas de performance vs configuraciones

# % 1 graficas de Performance vs configuraciones
plt.figure(figsize=(14, 8))  # Aumentado de (10, 6) a (14, 8)
plt.bar(folder_names, performance_values, color='skyblue')
plt.xlabel('Configuraciones', fontsize=14)
plt.ylabel('Performance (1/simSeconds)', fontsize=14)
plt.title('Performance vs Configuraciones', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.grid()
plt.savefig('graficos/performance_vs_configuraciones.svg', format='svg', bbox_inches='tight')
plt.show()


##% 2 graficas de EDP vs configuraciones
plt.figure(figsize=(14, 8))  # Aumentado de (10, 6) a (14, 8)
plt.bar(folder_names, edp_values, color='salmon')
plt.xlabel('Configuraciones', fontsize=14)
plt.ylabel('EDP', fontsize=14)
plt.title('EDP vs Configuraciones', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.grid()
plt.savefig('graficos/edp_vs_configuraciones.svg', format='svg', bbox_inches='tight')
plt.show()


# Encontrar óptimos de Pareto
pareto_mask = find_pareto_optimal(cpi_values, energy_values)
pareto_cpi = [cpi for i, cpi in enumerate(cpi_values) if pareto_mask[i]]
pareto_energy = [energy for i, energy in enumerate(energy_values) if pareto_mask[i]]
pareto_folders = [folder for i, folder in enumerate(folder_names) if pareto_mask[i]]

print(f"🔍 Óptimos de Pareto encontrados: {sum(pareto_mask)}/{len(cpi_values)}")

plt.figure(figsize=(15, 10))  # Aumentado de (12, 8) a (15, 10)

# Graficar todos los puntos (no óptimos)
non_pareto_cpi = [cpi for i, cpi in enumerate(cpi_values) if not pareto_mask[i]]
non_pareto_energy = [energy for i, energy in enumerate(energy_values) if not pareto_mask[i]]

plt.scatter(non_pareto_cpi, non_pareto_energy, color='lightgray', s=80, alpha=0.6, 
           label='Configuraciones no óptimas')

# Graficar óptimos de Pareto
plt.scatter(pareto_cpi, pareto_energy, color='red', s=120, alpha=0.8, 
           label='Óptimos de Pareto', edgecolors='black', linewidth=2)

# Conectar óptimos de Pareto con líneas
if len(pareto_cpi) > 1:
    # Ordenar por CPI para una frontera bonita
    sorted_indices = np.argsort(pareto_cpi)
    sorted_pareto_cpi = [pareto_cpi[i] for i in sorted_indices]
    sorted_pareto_energy = [pareto_energy[i] for i in sorted_indices]
    plt.plot(sorted_pareto_cpi, sorted_pareto_energy, 'r--', alpha=0.5, linewidth=2)

# Añadir etiquetas a los óptimos de Pareto
for i, (cpi, energy, folder) in enumerate(zip(pareto_cpi, pareto_energy, pareto_folders)):
    plt.annotate(folder, (cpi, energy), xytext=(5, 5), textcoords='offset points',
                fontsize=10, alpha=0.8, color='darkred')

plt.xlabel('CPI (menor es mejor)', fontsize=14)
plt.ylabel('Energy (J) (menor es mejor)', fontsize=14)
plt.title('Energy vs CPI - Frontera de Pareto\n(Minimizar ambos objetivos)', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('graficos/energy_vs_cpi_pareto.svg', format='svg', bbox_inches='tight')
plt.show()

# Mostrar información de los óptimos de Pareto
print(f"\n🏆 CONFIGURACIONES ÓPTIMAS DE PARETO:")
for i, (folder, cpi, energy) in enumerate(zip(pareto_folders, pareto_cpi, pareto_energy)):
    print(f"  {i+1}. {folder}")
    print(f"     CPI: {cpi:.4f}, Energy: {energy:.4f} J")