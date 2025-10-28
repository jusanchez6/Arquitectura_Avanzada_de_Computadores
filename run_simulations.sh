#!/bin/bash

WORKLOAD="mp3dec"

for fu in 2 4 8; do
    for sq in 72 144 288; do  
        for l1d in 32kB 64kB 128kB; do
            OUTDIR="results/${WORKLOAD}_fu${fu}_sq${sq}_l1d${l1d}"
            mkdir -p ${OUTDIR}

            CMD="./build/ARM/gem5.opt --outdir=${OUTDIR} resources_uarch_sim_assignment/scripts/CortexA76_scripts_gem5/CortexA76.py --cmd=resources_uarch_sim_assignment/workloads/mp3_dec/mp3_dec --options='-w mp3dec_outfile.wav resources_uarch_sim_assignment/workloads/mp3_dec/mp3dec_testfile.mp3' --num_fu_FP_SIMD_ALU=${fu} --sq_entries=${sq} --l1d_size=${l1d} --l1i_size=32kB"

            echo "=== Simulation config: issue_width=${fu}, sq_entries=${sq}, L1D=${l1d} ==="
            echo "Command: ${CMD}"
            echo "Outdir: ${OUTDIR}"
            
            eval ${CMD}
            
            echo ""
        done
    done
done

echo "Total de combinaciones: $((3 * 3 * 3))"

