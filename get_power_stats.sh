#!/bin/bash

SCRIPT_PYTHON="resources_uarch_sim_assignment/scripts/McPAT/gem5toMcPAT_cortexA76.py" 
SCRIPT_XML="resources_uarch_sim_assignment/scripts/McPAT/ARM_A76_2.1GHz.xml"
WORKLOAD="mp3dec"

for fu in 2 4 8; do
    for sq in 72 144 288; do
        for l1d in 32kB 64kB 128kB; do
            DIR_PATH="results/${WORKLOAD}_fu${fu}_sq${sq}_l1d${l1d}"
            
            # Verificar que el directorio existe antes de continuar
            if [ ! -d "$DIR_PATH" ]; then
                echo "Error: Directorio no encontrado: $DIR_PATH"
                continue
            fi
            
            # Comando para generar config.xml - CORREGIDO
            get_config="python $SCRIPT_PYTHON ${DIR_PATH}/stats.txt ${DIR_PATH}/config.json $SCRIPT_XML"
            
            echo "== Generando config.xml para: FU=${fu}, SQ=${sq}, L1D=${l1d} ==="
            echo "Outdir: ${DIR_PATH}"
            eval $get_config
            
            # Verificar que config.xml se creó correctamente
            if [ ! -f "config.xml" ]; then
                echo "Error: config.xml no se generó correctamente"
                continue
            fi
            
            # Comando para calcular potencia - CORREGIDO
            get_mcpat="./mcpat/mcpat -infile config.xml -print_level 1 >> ${DIR_PATH}/power_results.txt"
            
            echo "== Calculando potencia =="
            eval $get_mcpat
            
            # Limpiar archivo temporal config.xml si es necesario
            rm -f config.xml
            
            echo ""
        done
    done
done
