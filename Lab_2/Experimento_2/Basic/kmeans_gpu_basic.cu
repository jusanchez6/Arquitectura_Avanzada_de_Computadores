%%writefile kmeans_optimized.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <cuda_runtime.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>

#define MAX_POINTS 1000000
#define MAX_CLUSTERS 20
#define THREADS_PER_BLOCK 1024

__device__ float dist2(float px, float py, float cx, float cy) {
    float dx = px - cx;
    float dy = py - cy;
    return dx*dx + dy*dy;
}

__global__ void assign_clusters_advanced(
    const float *points,
    const float *centroids,
    int *labels,
    int N, int K)
{
    __shared__ float shared_centroids[MAX_CLUSTERS * 2];

    // Cargar centroides en shared
    for (int i = threadIdx.x; i < K * 2; i += blockDim.x)
        shared_centroids[i] = centroids[i];
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float px = points[2 * idx];
    float py = points[2 * idx + 1];

    float best_dist = FLT_MAX;
    int best_c = -1;

    for (int c = 0; c < K; c++) {
        float cx = shared_centroids[2 * c];
        float cy = shared_centroids[2 * c + 1];
        float d = dist2(px, py, cx, cy);
        if (d < best_dist) {
            best_dist = d;
            best_c = c;
        }
    }

    labels[idx] = best_c;
}


__global__ void accumulate_partials(
    const float *points,
    const int *labels,
    float *d_partial_sums,   // blocks * (K*2)
    int   *d_partial_counts, // blocks * K
    int N, int K)
{
    extern __shared__ float shared_mem[]; // tamaño: K*2*sizeof(float) + K*sizeof(int)
    float *sums = shared_mem;
    int *counts = (int*)&sums[K * 2];

    // Inicializar sums y counts en shared
    for (int i = threadIdx.x; i < K * 2; i += blockDim.x)
        sums[i] = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x)
        counts[i] = 0;
    __syncthreads();

    // Procesar los puntos asignados a este bloque (stride grid)
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int idx = global_tid; idx < N; idx += stride) {
        int c = labels[idx];
        if (c >= 0 && c < K) {
            float x = points[2 * idx];
            float y = points[2 * idx + 1];
            // atomics en shared son más eficientes que atomics globales
            atomicAdd(&sums[2 * c], x);
            atomicAdd(&sums[2 * c + 1], y);
            atomicAdd(&counts[c], 1);
        }
    }
    __syncthreads();

    // Escribir parciales del bloque a memoria global
    int block_offset_f = blockIdx.x * (K * 2);
    int block_offset_i = blockIdx.x * K;

    // Escribir sums (float)
    for (int i = threadIdx.x; i < K * 2; i += blockDim.x) {
        d_partial_sums[block_offset_f + i] = sums[i];
    }
    // Escribir counts (int)
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        d_partial_counts[block_offset_i + i] = counts[i];
    }
}


__global__ void finalize_centroids(
    const float *d_partial_sums,
    const int   *d_partial_counts,
    float *d_centroids,
    int blocks,
    int K)
{
    // Cada thread se encarga de uno o varios clusters
    int tid = threadIdx.x;
    int stride = blockDim.x;

    for (int c = tid; c < K; c += stride) {
        double sumx = 0.0;
        double sumy = 0.0;
        int total = 0;
        // Sumar sobre todos los bloques
        int offset_f = 2 * c;
        for (int b = 0; b < blocks; b++) {
            int base_f = b * (K * 2);
            int base_i = b * K;
            sumx += (double)d_partial_sums[base_f + offset_f];
            sumy += (double)d_partial_sums[base_f + offset_f + 1];
            total += d_partial_counts[base_i + c];
        }
        if (total > 0) {
            d_centroids[2 * c]     = (float)(sumx / total);
            d_centroids[2 * c + 1] = (float)(sumy / total);
        }
        // Si total == 0, dejamos el centroide como estaba
    }
}


int load_csv(const char *filename, float **out_points) {
    FILE *f = fopen(filename, "r");
    if (!f) return -1;

    float *points = (float*)malloc(MAX_POINTS * 2 * sizeof(float));
    int count = 0;
    float x, y;

    while (fscanf(f, "%f,%f", &x, &y) == 2) {
        points[2*count] = x;
        points[2*count+1] = y;
        count++;
    }
    fclose(f);
    *out_points = points;
    return count;
}

void initialize_random_centroids(float *points, int N, float *centroids, int K) {
    srand(12345);
    for (int c = 0; c < K; c++) {
        int idx = rand() % N;
        centroids[2 * c] = points[2 * idx];
        centroids[2 * c + 1] = points[2 * idx + 1];

        printf("Centroide: %0.2f , %0.2f \n", points[2 * idx],  points[2 * idx + 1]);
    }
}

void print_results(float *centroids, int K, float execution_time, int iterations) {
    printf("\nFinal centroids:\n");
    for (int c = 0; c < K; c++) {
        printf("C%d = (%.4f, %.4f)\n", c, centroids[2 * c], centroids[2 * c + 1]);
    }

    printf("Execution time: %.2f ms\n", execution_time);

    if (iterations > 0) {
        float norm = execution_time / iterations;
        printf("Iterations: %d\n", iterations);
        printf("Normalized time per iteration: %.4f ms/iter\n", norm);
    } else {
        printf("Iterations: N/A\n");
        printf("Normalized time per iteration: N/A\n");
    }
}

void kmeans_advanced(
    float *d_points,
    float *d_centroids,
    int N,
    int K,
    int max_iters,
    float epsilon,
    int blocks,
    int threads,
    int *h_iterations)
{
    int *d_labels = nullptr;
    cudaMalloc(&d_labels, N * sizeof(int));

    // arrays para parciales por bloque
    float *d_partial_sums = nullptr;   // blocks * (K*2)
    int   *d_partial_counts = nullptr; // blocks * K

    size_t partial_sums_bytes = (size_t)blocks * (size_t)(K * 2) * sizeof(float);
    size_t partial_counts_bytes = (size_t)blocks * (size_t)K * sizeof(int);

    cudaMalloc(&d_partial_sums, partial_sums_bytes);
    cudaMalloc(&d_partial_counts, partial_counts_bytes);

    int *h_labels = (int*)malloc(N * sizeof(int));
    float *h_centroids = (float*)malloc(K * 2 * sizeof(float));
    float *h_old_centroids = (float*)malloc(K * 2 * sizeof(float));

    // shared mem size para accumulate_partials
    size_t shared_mem = (K * 2 * sizeof(float)) + (K * sizeof(int));

    for (int it = 0; it < max_iters; it++) {

        // guardar centroides previos en host
        cudaMemcpy(h_old_centroids, d_centroids, K * 2 * sizeof(float), cudaMemcpyDeviceToHost);

        // 1) asignar clusters
        assign_clusters_advanced<<<blocks, threads>>>(d_points, d_centroids, d_labels, N, K);
        cudaDeviceSynchronize();

        cudaMemcpy(h_labels, d_labels, N * sizeof(int), cudaMemcpyDeviceToHost);
        // save_labels_cpu(h_labels, N, it);

        // 2) acumular parciales por bloque (cada bloque escribe sus parciales)
        // inicializar parciales a 0 (no estrictamente necesario si cada bloque sobrescribe todas sus posiciones,
        // pero por seguridad lo inicializamos)
        cudaMemset(d_partial_sums, 0, partial_sums_bytes);
        cudaMemset(d_partial_counts, 0, partial_counts_bytes);

        accumulate_partials<<<blocks, threads, shared_mem>>>(
            d_points, d_labels, d_partial_sums, d_partial_counts, N, K);
        cudaDeviceSynchronize();

        // 3) reducir parciales globales y calcular centroides (kernel de 1 bloque)
        int threads_finalize = (threads > 128) ? 128 : threads; // suficiente, K <= MAX_CLUSTERS
        if (threads_finalize < 1) threads_finalize = 1;
        finalize_centroids<<<1, threads_finalize>>>(
            d_partial_sums, d_partial_counts, d_centroids, blocks, K);
        cudaDeviceSynchronize();

        // copiar centroides al host para criterio de convergencia
        cudaMemcpy(h_centroids, d_centroids, K * 2 * sizeof(float), cudaMemcpyDeviceToHost);

        // convergence check (distancia euclidiana entre old y new)
        float movement = 0.0f;
        for (int c = 0; c < K; c++) {
            float dx = h_centroids[2*c]     - h_old_centroids[2*c];
            float dy = h_centroids[2*c + 1] - h_old_centroids[2*c + 1];
            movement += dx*dx + dy*dy;
        }

        if (movement < epsilon) {
            printf("Convergencia en iteración %d\n", it);
            *h_iterations = it;
            break;
        }


    }

    free(h_labels);
    free(h_centroids);
    free(h_old_centroids);

    cudaFree(d_labels);
    cudaFree(d_partial_sums);
    cudaFree(d_partial_counts);
}

/* --- main similar a tu versión original --- */
int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s data.csv [K] [blocks] [threads]\n", argv[0]);
        return 1;
    }

    int K = (argc > 2) ? atoi(argv[2]) : 3;
    int blocks = (argc > 3) ? atoi(argv[3]) : 64;
    int threads = (argc > 4) ? atoi(argv[4]) : 1024;
    int max_iters = 10000;

    float *h_points = NULL;
    int iterations = 0;
    int N = load_csv(argv[1], &h_points);
    if (N <= 0) return 1;

    printf("Loaded %d points, K=%d\n", N, K);

    float *h_centroids = (float*)malloc(K * 2 * sizeof(float));
    initialize_random_centroids(h_points, N, h_centroids, K);

    float *d_points, *d_centroids;
    cudaMalloc(&d_points, N * 2 * sizeof(float));
    cudaMalloc(&d_centroids, K * 2 * sizeof(float));

    cudaMemcpy(d_points, h_points, N * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids, K * 2 * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    kmeans_advanced(d_points, d_centroids, N, K, max_iters, 1e-4f, blocks, threads, &iterations);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_centroids, d_centroids, K * 2 * sizeof(float), cudaMemcpyDeviceToHost);

    print_results(h_centroids, K, ms, iterations);

    free(h_centroids);
    free(h_points);
    cudaFree(d_points);
    cudaFree(d_centroids);

    printf("\nOptimized (fixed) version complete.\n");
    return 0;
}
