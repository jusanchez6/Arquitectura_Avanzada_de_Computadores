import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import numpy as np

# Configuración simple
plt.rcParams['figure.figsize'] = [10, 6]
sns.set_style("whitegrid")

# --- Cargar los datos ---
explored = pd.read_csv("explored_configs.csv")
results = pd.read_csv("annealing_results.csv")

# --- Calcular métricas según la guía ---
explored.columns = [c.strip() for c in explored.columns]

# Calcular Energy y EDP según la fórmula del assignment
explored["Energy"] = (explored["Total Leakage"] + explored["Runtime Dynamic"]) * explored["CPI"]
explored["EDP"] = explored["Energy"] * explored["CPI"]
explored["performance"] = 1 / explored["cost"]  # simSeconds^-1

print("=== Primeras filas de datos ===")
print(explored[['FP_SIMD_ALU', 'INT_ALU', 'WRITE', 'SQ', 'L1D', 'CPI', 'Energy', 'EDP', 'performance']].head())

# =====================================================
# 1. EVOLUCIÓN DEL RECOCIDO SIMULADO
# =====================================================

# Procesar resultados del annealing
results["cfg_dict"] = results["best_cfg"].apply(ast.literal_eval)

param_mapping = {
    "num_fu_FP_SIMD_ALU": "FP_SIMD_ALU",
    "num_fu_intALU": "INT_ALU", 
    "num_fu_write": "WRITE",
    "sq_entries": "SQ",
    "l1d_size": "L1D"
}

for k, v in param_mapping.items():
    results[k] = results["cfg_dict"].apply(lambda d: d.get(k))

def find_metrics(cfg):
    row = explored[
        (explored["FP_SIMD_ALU"] == cfg["num_fu_FP_SIMD_ALU"]) &
        (explored["INT_ALU"] == cfg["num_fu_intALU"]) &
        (explored["WRITE"] == cfg["num_fu_write"]) &
        (explored["SQ"] == cfg["sq_entries"]) &
        (explored["L1D"] == cfg["l1d_size"])
    ]
    if not row.empty:
        return {
            'EDP': row["EDP"].values[0],
            'performance': row["performance"].values[0],
            'Energy': row["Energy"].values[0]
        }
    return None

metrics_data = results["cfg_dict"].apply(find_metrics)
results["best_EDP"] = metrics_data.apply(lambda x: x['EDP'] if x else None)
results["best_performance"] = metrics_data.apply(lambda x: x['performance'] if x else None)
results["best_Energy"] = metrics_data.apply(lambda x: x['Energy'] if x else None)

# Normalizar las métricas para mejor visualización
results["norm_EDP"] = (results["best_EDP"] - results["best_EDP"].min()) / (results["best_EDP"].max() - results["best_EDP"].min())
results["norm_performance"] = (results["best_performance"] - results["best_performance"].min()) / (results["best_performance"].max() - results["best_performance"].min())

# Gráfico de evolución NORMALIZADO
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# EDP normalizado (queremos minimizar)
ax1.plot(results["round"], results["norm_EDP"], marker="o", color="red", linewidth=2)
ax1.set_title("Evolución del EDP (Normalizado)")
ax1.set_xlabel("Iteración")
ax1.set_ylabel("EDP Normalizado")
ax1.grid(True)
ax1.set_ylim(-0.1, 1.1)

# Performance normalizado (queremos maximizar)
ax2.plot(results["round"], results["norm_performance"], marker="s", color="blue", linewidth=2)
ax2.set_title("Evolución del Performance (Normalizado)")
ax2.set_xlabel("Iteración")
ax2.set_ylabel("Performance Normalizado")
ax2.grid(True)
ax2.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig("graficos/annealing_evolution_normalized.png", format='png', bbox_inches='tight', dpi=300)
plt.show()

# =====================================================
# 2. PERFORMANCE VS ENERGY (Requerimiento principal)
# =====================================================

# Normalizar Energy y Performance para el scatter plot
explored["norm_Energy"] = (explored['Energy'] - explored['Energy'].min()) / (explored['Energy'].max() - explored['Energy'].min())
explored["norm_performance"] = (explored['performance'] - explored['performance'].min()) / (explored['performance'].max() - explored['performance'].min())

# Función para encontrar el frente de Pareto
def find_pareto_front(df, cost_col, benefit_col):
    """
    Encuentra el frente de Pareto minimizando cost_col y maximizando benefit_col
    """
    pareto_mask = np.ones(len(df), dtype=bool)
    df_sorted = df.sort_values(by=[cost_col, benefit_col], ascending=[True, False])
    
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        if pareto_mask[idx]:
            # Un punto domina si tiene menor costo Y mayor beneficio
            dominated = (df[cost_col] <= row[cost_col]) & (df[benefit_col] >= row[benefit_col])
            dominated[idx] = False  # No se domina a sí mismo
            pareto_mask[dominated] = False
    return pareto_mask

# Encontrar frente de Pareto (minimizar Energy, maximizar performance)
pareto_mask = find_pareto_front(explored, 'norm_Energy', 'norm_performance')
pareto_points = explored[pareto_mask].sort_values('norm_Energy')

plt.figure(figsize=(10, 7))

# Todos los puntos explorados (normalizados)
plt.scatter(explored['norm_Energy'], explored['norm_performance'], 
           alpha=0.6, s=30, label='Todas las configuraciones', color='gray')

# Frente de Pareto
plt.plot(pareto_points['norm_Energy'], pareto_points['norm_performance'], 
        'r-', linewidth=2, label='Frente de Pareto', alpha=0.8)

# Encontrar las 3 mejores configuraciones según lo pedido
best_performance = explored.loc[explored['norm_performance'].idxmax()]
best_energy = explored.loc[explored['norm_Energy'].idxmin()]
best_edp = explored.loc[explored['EDP'].idxmin()]

# Destacar las 3 mejores configuraciones
plt.scatter(best_performance['norm_Energy'], best_performance['norm_performance'], 
           s=150, marker='*', color='red', label='Mejor Performance', edgecolors='black')
plt.scatter(best_energy['norm_Energy'], best_energy['norm_performance'], 
           s=150, marker='D', color='green', label='Mejor Energy', edgecolors='black')
plt.scatter(best_edp['norm_Energy'], best_edp['norm_performance'], 
           s=150, marker='s', color='orange', label='Mejor EDP', edgecolors='black')

plt.xlabel("Energy Normalizado (0 = mejor, 1 = peor)")
plt.ylabel("Performance Normalizado (0 = peor, 1 = mejor)")
plt.title("Performance vs Energy - Frente de Pareto")
plt.legend()
plt.grid(True, alpha=0.3)

# Agregar anotaciones con valores reales
plt.annotate(f'Perf: {best_performance["performance"]:.4f}\nEnergy: {best_performance["Energy"]:.4f}', 
             xy=(best_performance['norm_Energy'], best_performance['norm_performance']),
             xytext=(10, 10), textcoords='offset points', fontsize=9)
plt.annotate(f'Perf: {best_energy["performance"]:.4f}\nEnergy: {best_energy["Energy"]:.4f}', 
             xy=(best_energy['norm_Energy'], best_energy['norm_performance']),
             xytext=(10, 10), textcoords='offset points', fontsize=9)
plt.annotate(f'Perf: {best_edp["performance"]:.4f}\nEnergy: {best_edp["Energy"]:.4f}', 
             xy=(best_edp['norm_Energy'], best_edp['norm_performance']),
             xytext=(10, 10), textcoords='offset points', fontsize=9)

plt.tight_layout()
plt.savefig("graficos/performance_vs_energy_pareto.png", format='png', bbox_inches='tight', dpi=300)
plt.show()

# =====================================================
# IMPRIMIR INFORMACIÓN DEL FRENTE DE PARETO PARA EL INFORME
# =====================================================

print("="*70)
print("ANÁLISIS DEL FRENTE DE PARETO")
print("="*70)

print(f"\nESTADÍSTICAS DEL FRENTE DE PARETO:")
print(f"   Total de configuraciones en el frente de Pareto: {len(pareto_points)}")
print(f"   Porcentaje del espacio explorado: {len(pareto_points)/len(explored)*100:.1f}%")

print(f"\nMEJORES CONFIGURACIONES PARETO-ÓPTIMAS:")

# Mostrar las 5 mejores configuraciones del frente de Pareto
print(f"\nTop 5 configuraciones Pareto-óptimas (ordenadas por Performance):")
top_pareto = pareto_points.nlargest(5, 'norm_performance')[['FP_SIMD_ALU', 'INT_ALU', 'WRITE', 'SQ', 'L1D', 
                                                           'performance', 'Energy', 'EDP']]
for i, (idx, row) in enumerate(top_pareto.iterrows(), 1):
    print(f"\n{i}. FP_SIMD_ALU={row['FP_SIMD_ALU']}, INT_ALU={row['INT_ALU']}, WRITE={row['WRITE']}, "
          f"SQ={row['SQ']}, L1D={row['L1D']}")
    print(f"   Performance: {row['performance']:.6f}")
    print(f"   Energy: {row['Energy']:.6f}")
    print(f"   EDP: {row['EDP']:.6f}")

print(f"\nRANGOS EN EL FRENTE DE PARETO:")
print(f"   Performance: {pareto_points['performance'].min():.6f} - {pareto_points['performance'].max():.6f}")
print(f"   Energy: {pareto_points['Energy'].min():.6f} - {pareto_points['Energy'].max():.6f}")
print(f"   EDP: {pareto_points['EDP'].min():.6f} - {pareto_points['EDP'].max():.6f}")

print(f"\nCONFIGURACIONES ÓPTIMAS ESPECÍFICAS:")
print(f"   Mejor Performance absoluto: FP_SIMD_ALU={best_performance['FP_SIMD_ALU']}, "
      f"INT_ALU={best_performance['INT_ALU']}")
print(f"   Mejor Energy absoluto: FP_SIMD_ALU={best_energy['FP_SIMD_ALU']}, "
      f"INT_ALU={best_energy['INT_ALU']}")
print(f"   Mejor EDP absoluto: FP_SIMD_ALU={best_edp['FP_SIMD_ALU']}, "
      f"INT_ALU={best_edp['INT_ALU']}")

# Verificar si las mejores configuraciones están en el frente de Pareto
print(f"\nVERIFICACIÓN PARETO:")
print(f"   Mejor Performance en Pareto: {pareto_mask[best_performance.name]}")
print(f"   Mejor Energy en Pareto: {pareto_mask[best_energy.name]}")
print(f"   Mejor EDP en Pareto: {pareto_mask[best_edp.name]}")

print("="*70)
# =====================================================
# 3. ANÁLISIS DE PARÁMETROS INDIVIDUALES
# =====================================================

parameters = ['FP_SIMD_ALU', 'INT_ALU', 'WRITE', 'SQ', 'L1D']
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, param in enumerate(parameters):
    if i < len(axes):
        grouped = explored.groupby(param).agg({
            'performance': 'mean',
            'EDP': 'mean',
            'Energy': 'mean'
        }).reset_index()
        
        # Normalizar los valores para mejor comparación
        grouped['norm_performance'] = (grouped['performance'] - grouped['performance'].min()) / (grouped['performance'].max() - grouped['performance'].min())
        grouped['norm_EDP'] = (grouped['EDP'] - grouped['EDP'].min()) / (grouped['EDP'].max() - grouped['EDP'].min())
        
        ax = axes[i]
        ax.plot(grouped[param], grouped['norm_performance'], 'o-', color='blue', linewidth=2, label='Performance')
        ax.set_xlabel(param)
        ax.set_ylabel('Performance Normalizado', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.set_ylim(-0.1, 1.1)
        
        ax2 = ax.twinx()
        ax2.plot(grouped[param], grouped['norm_EDP'], 's-', color='red', linewidth=2, label='EDP')
        ax2.set_ylabel('EDP Normalizado', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(-0.1, 1.1)
        
        # Agregar leyenda combinada
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax.set_title(f'Efecto de {param}')
        ax.grid(True, alpha=0.3)

# Eliminar ejes vacíos
for i in range(len(parameters), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig("graficos/param_effects_normalized.png", format='png', bbox_inches='tight', dpi=300)
plt.show()

# Mostrar valores de normalización para referencia
print("Rangos de normalización por parámetro:")
for param in parameters:
    grouped = explored.groupby(param).agg({
        'performance': ['min', 'max'],
        'EDP': ['min', 'max']
    }).reset_index()
    print(f"\n{param}:")
    print(f"  Performance: {grouped['performance']['min'].min():.6f} - {grouped['performance']['max'].max():.6f}")
    print(f"  EDP: {grouped['EDP']['min'].min():.6f} - {grouped['EDP']['max'].max():.6f}")

# =====================================================
# 4. RESUMEN EJECUTIVO PARA EL REPORTE
# =====================================================

print("="*60)
print("RESULTADOS PARA EL REPORTE IEEE")
print("="*60)

print("\nTRES MEJORES CONFIGURACIONES (según lo requerido):")

print(f"\n1. MEJOR PERFORMANCE:")
print(f"   Configuración: FP_SIMD_ALU={best_performance['FP_SIMD_ALU']}, "
      f"INT_ALU={best_performance['INT_ALU']}, WRITE={best_performance['WRITE']}, "
      f"SQ={best_performance['SQ']}, L1D={best_performance['L1D']}")
print(f"   Performance: {best_performance['performance']:.6f}")
print(f"   Energy: {best_performance['Energy']:.6f}")
print(f"   EDP: {best_performance['EDP']:.6f}")

print(f"\n2. MEJOR ENERGY:")
print(f"   Configuración: FP_SIMD_ALU={best_energy['FP_SIMD_ALU']}, "
      f"INT_ALU={best_energy['INT_ALU']}, WRITE={best_energy['WRITE']}, "
      f"SQ={best_energy['SQ']}, L1D={best_energy['L1D']}")
print(f"   Performance: {best_energy['performance']:.6f}")
print(f"   Energy: {best_energy['Energy']:.6f}")
print(f"   EDP: {best_energy['EDP']:.6f}")

print(f"\n3. MEJOR EDP:")
print(f"   Configuración: FP_SIMD_ALU={best_edp['FP_SIMD_ALU']}, "
      f"INT_ALU={best_edp['INT_ALU']}, WRITE={best_edp['WRITE']}, "
      f"SQ={best_edp['SQ']}, L1D={best_edp['L1D']}")
print(f"   Performance: {best_edp['performance']:.6f}")
print(f"   Energy: {best_edp['Energy']:.6f}")
print(f"   EDP: {best_edp['EDP']:.6f}")

# Efectividad del recocido simulado
if len(results) > 0:
    initial_edp = results['best_EDP'].iloc[0]
    final_edp = results['best_EDP'].iloc[-1]
    improvement = (initial_edp - final_edp) / initial_edp * 100
    
    print(f"\nEFECTIVIDAD DEL RECOCIDO SIMULADO:")
    print(f"   EDP inicial: {initial_edp:.6f}")
    print(f"   EDP final: {final_edp:.6f}")
    print(f"   Mejora: {improvement:.1f}%")

print(f"\nESTADÍSTICAS GENERALES:")
print(f"   Total configuraciones exploradas: {len(explored)}")
print(f"   Rango de Performance: {explored['performance'].min():.4f} - {explored['performance'].max():.4f}")
print(f"   Rango de Energy: {explored['Energy'].min():.4f} - {explored['Energy'].max():.4f}")
print(f"   Rango de EDP: {explored['EDP'].min():.4f} - {explored['EDP'].max():.4f}")

print("="*60)