%%writefile kmeans_optimized.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <cuda_runtime.h>
#include <time.h>

#define MAX_POINTS 100000
#define MAX_CLUSTERS 20
#define THREADS_PER_BLOCK 256

__device__ float dist2(float px, float py, float cx, float cy) {
    float dx = px - cx;
    float dy = py - cy;
    return (dx * dx) + (dy * dy);
}

__global__ void assign_clusters_advanced(
    const float *points,
    const float *centroids,
    int *labels,
    int N, int K) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float px = points[2 * idx];
    float py = points[2 * idx + 1];

    float best_dist = FLT_MAX;
    int best_c = -1;

    // OPTIMIZACIÓN
    __shared__ float shared_centroids[MAX_CLUSTERS * 2];
    for (int i = threadIdx.x; i < K * 2; i += blockDim.x) {
        if (i < K * 2) {
            shared_centroids[i] = centroids[i];
        }
    }
    __syncthreads();

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

__global__ void update_centroids_advanced(
    const float *points, 
    float *centroids,
    const int *labels,
    int N, 
    int K
){
    extern __shared__ float shared_data[];
    float *sums = shared_data;
    int *counts = (int*)&sums[K * 2];

    // incializar los vectores
    for (int i = threadIdx.x; i < K * 2; i += blockDim.x) {
        sums[i] = 0.0f;
    }
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        counts[i] = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        int c = labels[idx];
        float x = points[2 * idx];
        float y = points[2 * idx + 1];

        atomicAdd(&sums[2 * c], x);
        atomicAdd(&sums[2 * c + 1], y);
        atomicAdd(&counts[c], 1);
    }
    __syncthreads();

    if (threadIdx.x < K) {
        if (counts[threadIdx.x] > 0) {
            float new_x = sums[2 * threadIdx.x] / counts[threadIdx.x];
            float new_y = sums[2 * threadIdx.x + 1] / counts[threadIdx.x];

            centroids[2 * threadIdx.x] = new_x;
            centroids[2 * threadIdx.x + 1] = new_y;
        }
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
    int threads
){
    int *d_labels;
    cudaMalloc(&d_labels, N * sizeof(int));

    size_t shared_mem_size = (K * 2 * sizeof(float)) + (K * sizeof(int));

    printf("Using shared memory: %zu bytes\n", shared_mem_size);

    for (int it = 0; it < max_iters; it++) {
        assign_clusters_advanced<<<blocks, threads>>>(d_points, d_centroids, d_labels, N, K);
        cudaDeviceSynchronize();

        update_centroids_advanced<<<blocks, threads, shared_mem_size>>>(d_points, d_centroids, d_labels, N, K);
        cudaDeviceSynchronize();

        printf("Iteration %d completed.\n", it);
        
        if (it >= 5) {
            break;
        }
    }

    cudaFree(d_labels);
}



int load_csv(const char *filename, float **out_points) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        printf("Error: cannot open file %s\n", filename);
        return -1;
    }

    float *points = (float*)malloc(MAX_POINTS * 2 * sizeof(float));
    if (!points) {
        printf("Error allocating memory\n");
        fclose(f);
        return -1;
    }

    int count = 0;
    float x, y;

    while (fscanf(f, "%f,%f", &x, &y) == 2) {
        if (count >= MAX_POINTS) break;
        points[2 * count] = x;
        points[2 * count + 1] = y;
        count++;
    }

    fclose(f);
    *out_points = points;
    return count;
}

void initialize_centroids_simple(float *centroids, int K) {
    srand(time(NULL));
    for (int c = 0; c < K; c++) {
        centroids[2 * c] = (rand() % 200 - 100) / 10.0f; // -10 to 10
        centroids[2 * c + 1] = (rand() % 200 - 100) / 10.0f;
    }
}

void print_results(float *centroids, int K, float execution_time) {
    printf("\nFinal centroids:\n");
    for (int c = 0; c < K; c++) {
        printf("C%d = (%.4f, %.4f)\n", c, centroids[2 * c], centroids[2 * c + 1]);
    }
    printf("Execution time: %.2f ms\n", execution_time);
}


int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s data.csv [K] [blocks] [threads]\n", argv[0]);
        printf("Example: %s small_input.csv 3 64 256\n", argv[0]);
        return 1;
    }

    // Parámetros
    int K = (argc > 2) ? atoi(argv[2]) : 3;
    int blocks = (argc > 3) ? atoi(argv[3]) : 64;
    int threads = (argc > 4) ? atoi(argv[4]) : 256;
    int max_iters = 10;
    float epsilon = 1e-6f;

    // Validar K
    if (K > MAX_CLUSTERS) {
        printf("Error: K=%d exceeds MAX_CLUSTERS=%d\n", K, MAX_CLUSTERS);
        return 1;
    }

    // Cargar datos
    float *h_points = NULL;
    int N = load_csv(argv[1], &h_points);
    if (N <= 0) {
        printf("Error loading data from %s\n", argv[1]);
        return 1;
    }

    printf("=== OPTIMIZED K-MEANS GPU VERSION ===\n");
    printf("Loaded %d points, K=%d\n", N, K);
    printf("Configuration: %d blocks, %d threads\n", blocks, threads);

    // Memoria dinámica para centroides
    float *h_centroids = (float*)malloc(K * 2 * sizeof(float));
    if (!h_centroids) {
        printf("Error allocating memory for centroids\n");
        free(h_points);
        return 1;
    }

    // Memoria GPU
    float *d_points, *d_centroids;
    cudaMalloc(&d_points, N * 2 * sizeof(float));
    cudaMalloc(&d_centroids, K * 2 * sizeof(float));

    // Copiar datos a GPU
    cudaMemcpy(d_points, h_points, N * 2 * sizeof(float), cudaMemcpyHostToDevice);

    // Inicializar centroides
    initialize_centroids_simple(h_centroids, K);

    printf("Initial centroids:\n");
    for (int c = 0; c < K; c++) {
        printf("C%d = (%.4f, %.4f)\n", c, h_centroids[2 * c], h_centroids[2 * c + 1]);
    }

    cudaMemcpy(d_centroids, h_centroids, K * 2 * sizeof(float), cudaMemcpyHostToDevice);

    // Medir tiempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("\n=== Running Optimized K-means ===\n");

    cudaEventRecord(start);
    
    // Ejecutar versión optimizada
    kmeans_advanced(d_points, d_centroids, N, K, max_iters, epsilon, blocks, threads);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copiar resultados
    cudaMemcpy(h_centroids, d_centroids, K * 2 * sizeof(float), cudaMemcpyDeviceToHost);

    // Resultados
    print_results(h_centroids, K, milliseconds);

    // Limpiar
    free(h_centroids);
    free(h_points);
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\nOptimized version completed!\n");
    return 0;
}