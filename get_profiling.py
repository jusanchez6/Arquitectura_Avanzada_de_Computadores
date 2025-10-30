import os
import re
import matplotlib.pyplot as plt
import numpy as np

# ===== CONFIGURACIÓN DE FONT Y TAMAÑOS =====
# Configurar el font size globalmente
plt.rcParams.update({
    'font.size': 14,           # Tamaño base del font
    'axes.titlesize': 18,      # Tamaño de los títulos
    'axes.labelsize': 16,      # Tamaño de las etiquetas de ejes
    'xtick.labelsize': 14,     # Tamaño de las etiquetas del eje X
    'ytick.labelsize': 14,     # Tamaño de las etiquetas del eje Y
    'legend.fontsize': 14,     # Tamaño de la leyenda
    'figure.titlesize': 20     # Tamaño del título de la figura
})

BASE_DIR = "results-profiling"

METRICAS_TIPO = {
    "Int": "numIntInsts",
    "Float": "numFpInsts", 
    "Load": "numLoadInsts",
    "Store": "numStoreInsts",
    "Vector": "numVecInsts",
}

# Mapeo de tipos de instrucciones a las categorías de statFuBusy
MAPEO_FU_BUSY = {
    "Int": [
        "IntAlu", "IntMult", "IntDiv"
    ],
    "Float": [
        "FloatAdd", "FloatCmp", "FloatCvt", "FloatMult", "FloatMultAcc", 
        "FloatDiv", "FloatMisc", "FloatSqrt", "FloatMemRead", "FloatMemWrite"
    ],
    "Load": [
        "MemRead", "FloatMemRead", "SimdUnitStrideLoad", "SimdUnitStrideMaskLoad",
        "SimdStridedLoad", "SimdIndexedLoad", "SimdWholeRegisterLoad",
        "SimdUnitStrideFaultOnlyFirstLoad", "SimdUnitStrideSegmentedLoad",
        "SimdUnitStrideSegmentedFaultOnlyFirstLoad", "SimdStrideSegmentedLoad"
    ],
    "Store": [
        "MemWrite", "FloatMemWrite", "SimdUnitStrideStore", "SimdUnitStrideMaskStore",
        "SimdStridedStore", "SimdIndexedStore", "SimdWholeRegisterStore",
        "SimdUnitStrideSegmentedStore", "SimdStrideSegmentedStore"
    ],
    "Vector": [
        "SimdAdd", "SimdAddAcc", "SimdAlu", "SimdCmp", "SimdCvt", "SimdMisc",
        "SimdMult", "SimdMultAcc", "SimdMatMultAcc", "SimdShift", "SimdShiftAcc",
        "SimdDiv", "SimdSqrt", "SimdFloatAdd", "SimdFloatAlu", "SimdFloatCmp",
        "SimdFloatCvt", "SimdFloatDiv", "SimdFloatMisc", "SimdFloatMult",
        "SimdFloatMultAcc", "SimdFloatMatMultAcc", "SimdFloatSqrt",
        "SimdReduceAdd", "SimdReduceAlu", "SimdReduceCmp", "SimdFloatReduceAdd",
        "SimdFloatReduceCmp", "SimdAes", "SimdAesMix", "SimdSha1Hash",
        "SimdSha1Hash2", "SimdSha256Hash", "SimdSha256Hash2", "SimdShaSigma2",
        "SimdShaSigma3", "SimdPredAlu", "Matrix", "MatrixMov", "MatrixOP",
        "SimdExt", "SimdFloatExt", "SimdConfig"
    ]
}

def extraer_tipos_instrucciones(path_stats):
    data = {}
    ocupadas_data = {}
    try:
        with open(path_stats) as f:
            contenido = f.read()

        # Extraer conteos normales de instrucciones
        for tipo, clave in METRICAS_TIPO.items():
            match = re.search(rf"{clave}\s+(\d+)", contenido)
            data[tipo] = int(match.group(1)) if match else 0

        # Extraer stalls por unidades funcionales ocupadas
        for tipo, categorias in MAPEO_FU_BUSY.items():
            total_ocupadas = 0
            for categoria in categorias:
                # Buscar patron: system.cpu.statFuBusy::Categoria valor
                pattern = rf"system\.cpu\.statFuBusy::{categoria}\s+(\d+)"
                match = re.search(pattern, contenido)
                if match:
                    total_ocupadas += int(match.group(1))
            ocupadas_data[tipo] = total_ocupadas

    except FileNotFoundError:
        print(f"[ERROR] No se encontró: {path_stats}")
        return None, None

    return data, ocupadas_data

def main():
    sims = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    resultados = {}
    resultados_ocupadas = {}

    for sim in sims:
        path_stats = os.path.join(BASE_DIR, sim, "stats.txt")
        tipos, ocupadas = extraer_tipos_instrucciones(path_stats)

        if tipos and any(v > 0 for v in tipos.values()):
            resultados[sim] = tipos
            resultados_ocupadas[sim] = ocupadas
        else:
            print(f"[WARN] {sim}: no se encontraron tipos de instrucciones válidos")

    if not resultados:
        print("\nNo se encontraron datos válidos para graficar.")
        return

    # === PRIMERA FIGURA: Profiling normal de instrucciones ===
    print("\n=== Profiling detallado por tipo de instrucción ===\n")
    for sim, tipos in resultados.items():
        total = sum(tipos.values())
        print(f"Simulación: {sim}  (Total: {total:,})")
        for tipo, valor in tipos.items():
            pct = (valor / total * 100) if total > 0 else 0
            print(f"  {tipo:<10} {valor:>12,} ({pct:>5.2f}%)")
        print()

    categorias = list(METRICAS_TIPO.keys())
    x = np.arange(len(categorias))
    ancho = 0.15

    # Figura 1: Instrucciones normales
    fig1, ax1 = plt.subplots(figsize=(16, 10))  # Aumenté el tamaño de la figura

    for i, (sim, tipos) in enumerate(resultados.items()):
        total = sum(tipos.values())
        valores = [tipos.get(cat, 0) for cat in categorias]
        porcentajes = [(v / total * 100) if total > 0 else 0 for v in valores]

        bars = ax1.bar(x + i * ancho, valores, width=ancho, label=sim)

        for bar, pct in zip(bars, porcentajes):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=12,  # Aumentado de 10.5 a 12
                rotation=0,
            )

    ax1.set_title("Profiling types of instructions", fontsize=20, pad=20)  # Título más grande
    ax1.set_xlabel("Instruction type", fontsize=16)
    ax1.set_ylabel("Instruction count", fontsize=16)
    ax1.set_xticks(x + ancho * (len(resultados) - 1) / 2)
    ax1.set_xticklabels(categorias, fontsize=14)
    ax1.legend(fontsize=14)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("graficas/profiling.svg", format="svg", bbox_inches="tight")
    plt.show()

    # === SEGUNDA FIGURA: Unidades funcionales ocupadas ===
    print("\n=== Stalls por unidades funcionales ocupadas ===\n")
    for sim, ocupadas in resultados_ocupadas.items():
        total_stalls = sum(ocupadas.values())
        total_instrucciones = sum(resultados[sim].values())
        print(f"Simulación: {sim}  (Total stalls: {total_stalls:,})")
        
        for tipo, valor in ocupadas.items():
            pct_stalls = (valor / total_stalls * 100) if total_stalls > 0 else 0
            pct_instrucciones = (valor / total_instrucciones * 100) if total_instrucciones > 0 else 0
            print(f"  {tipo:<10} {valor:>12,} stalls ({pct_stalls:>5.2f}% of stalls, {pct_instrucciones:>5.2f}% of total inst)")
        print()

    # Figura 2: Stalls por unidades funcionales ocupadas (valores absolutos)
    fig2, ax2 = plt.subplots(figsize=(16, 10))

    for i, (sim, ocupadas) in enumerate(resultados_ocupadas.items()):
        total_stalls = sum(ocupadas.values())
        valores = [ocupadas.get(cat, 0) for cat in categorias]
        porcentajes = [(v / total_stalls * 100) if total_stalls > 0 else 0 for v in valores]

        bars = ax2.bar(x + i * ancho, valores, width=ancho, label=sim)

        for bar, pct in zip(bars, porcentajes):
            height = bar.get_height()
            if height > 0:  # Solo mostrar etiqueta si hay valores
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{pct:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=12,  # Aumentado
                    rotation=0,
                )

    ax2.set_title("Functional Unit Occupancy Stalls by Instruction Type", fontsize=20, pad=20)
    ax2.set_xlabel("Instruction type", fontsize=16)
    ax2.set_ylabel("Number of stalls", fontsize=16)
    ax2.set_xticks(x + ancho * (len(resultados_ocupadas) - 1) / 2)
    ax2.set_xticklabels(categorias, fontsize=14)
    ax2.legend(fontsize=14)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("graficas/fu_occupancy_stalls.svg", format="svg", bbox_inches="tight")
    plt.show()

    # === TERCERA FIGURA: Porcentaje de stalls respecto al total de instrucciones ===
    fig3, ax3 = plt.subplots(figsize=(16, 10))

    for i, (sim, ocupadas) in enumerate(resultados_ocupadas.items()):
        total_instrucciones = sum(resultados[sim].values())
        porcentajes_stalls = [(ocupadas.get(cat, 0) / total_instrucciones * 100) 
                             if total_instrucciones > 0 else 0 
                             for cat in categorias]

        bars = ax3.bar(x + i * ancho, porcentajes_stalls, width=ancho, label=sim)

        for bar, pct in zip(bars, porcentajes_stalls):
            height = bar.get_height()
            if height > 0.1:  # Solo mostrar etiqueta si es significativo
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{pct:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=12,  # Aumentado
                    rotation=0,
                )

    ax3.set_title("Percentage of Instructions that Found Functional Units Occupied", fontsize=20, pad=20)
    ax3.set_xlabel("Instruction type", fontsize=16)
    ax3.set_ylabel("Percentage of total instructions (%)", fontsize=16)
    ax3.set_xticks(x + ancho * (len(resultados_ocupadas) - 1) / 2)
    ax3.set_xticklabels(categorias, fontsize=14)
    ax3.legend(fontsize=14)
    ax3.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("graficas/fu_occupancy_percentage.svg", format="svg", bbox_inches="tight")
    plt.show()

    # === FIGURA ADICIONAL: Ratio de stalls por tipo de instrucción ===
    fig4, ax4 = plt.subplots(figsize=(16, 10))

    for i, (sim, ocupadas) in enumerate(resultados_ocupadas.items()):
        ratios = []
        for cat in categorias:
            total_inst_tipo = resultados[sim].get(cat, 0)
            stalls_tipo = ocupadas.get(cat, 0)
            ratio = (stalls_tipo / total_inst_tipo * 100) if total_inst_tipo > 0 else 0
            ratios.append(ratio)

        bars = ax4.bar(x + i * ancho, ratios, width=ancho, label=sim)

        for bar, ratio in zip(bars, ratios):
            height = bar.get_height()
            if height > 0.1:  # Solo mostrar etiqueta si es significativo
                ax4.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{ratio:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=12,  # Aumentado
                    rotation=0,
                )

    ax4.set_title("Stall Ratio by Instruction Type (Stalls / Instructions of that type)", fontsize=20, pad=20)
    ax4.set_xlabel("Instruction type", fontsize=16)
    ax4.set_ylabel("Stall ratio (%)", fontsize=16)
    ax4.set_xticks(x + ancho * (len(resultados_ocupadas) - 1) / 2)
    ax4.set_xticklabels(categorias, fontsize=14)
    ax4.legend(fontsize=14)
    ax4.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("graficas/fu_occupancy_ratio.svg", format="svg", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()