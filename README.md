# Advanced Arquitecture Computer

Repo use to show the results for the first laboratory of AAC 

---

# Desing Space Exploration - DSE 

The following table shows the choosen DSE for the simulation:

| **Category**         | **Parameter**           | **Values**          | **Measurement Objective**                                                |
| -------------------- | ----------------------- | ------------------- | ------------------------------------------------------------------------ |
| **Caches**           | `l1d_assoc`             | [4, 8, 16]          | Measure how L1D associativity impacts CPI and energy consumption         |
|                      | `l2_size`               | [256kB, 512kB, 1MB] | Evaluate the effect of L2 cache size on memory latency and total energy  |
| **Pipeline**         | `rob_entries`           | [128, 192, 256]     | Analyze the influence of ROB size on performance (CPI) and power usage   |
|                      | `issue_width`           | [4, 8]              | Measure how pipeline width affects instruction throughput and EDP        |
| **Functional Units** | `num_fu_intALU`         | [2, 4]              | Observe the trade-off between parallel execution performance and power   |
| **Predictor**        | `branch_predictor_type` | [7, 10]             | Compare predictor types in terms of accuracy, CPI, and energy efficiency |

