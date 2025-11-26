%%writefile kmeans_gpu.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAX_POINTS 10000000
#define MAX_CLUSTERS 20
#define THREADS_PER_BLOCK 256

// Declaración de las funciones del kernel
__global__ void assign_clusters_basic(
    const float *points,
    const float *centroids,
    int *labels,
    int N, int K);

__global__ void update_centroids_basic(
    const float *points,
    float *centroids,
    const int *labels,
    float *sums,
    int *counts,
    int N, int K);

void kmeans_gpu_basic(
    float *d_points,
    float *d_centroids,
    int N, int K,
    int max_iters,
    float epsilon,
    int blocks,
    int threads);

// Función para cargar datos desde CSV
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

// Función para imprimir resultados
void print_results(float *centroids, int K, float execution_time) {
    printf("\nFinal centroids:\n");
    for (int c = 0; c < K; c++) {
        printf("C%d = (%.4f, %.4f)\n", c, centroids[2 * c], centroids[2 * c + 1]);
    }
    printf("Execution time: %.2f ms\n", execution_time);
}

void initialize_random_centroids(float *points, int N, float *centroids, int K) {
    // Semilla para números aleatorios
    srand(time(NULL));
    
    for (int c = 0; c < K; c++) {
        // Escoger un punto aleatorio como centroide inicial
        int random_idx = rand() % N;
        centroids[2 * c] = points[2 * random_idx];
        centroids[2 * c + 1] = points[2 * random_idx + 1];
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s data.csv [K] [blocks] [threads]\n", argv[0]);
        printf("Example: %s small_input.csv 3 64 256\n", argv[0]);
        return 1;
    }

    // Parámetros por defecto
    int K = (argc > 2) ? atoi(argv[2]) : 3;
    int blocks = (argc > 3) ? atoi(argv[3]) : 64;
    int threads = (argc > 4) ? atoi(argv[4]) : 256;
    int max_iters = 20;
    float epsilon = 1e-6f;

    // Cargar datos
    float *h_points = NULL;
    int N = load_csv(argv[1], &h_points);
    if (N <= 0) {
        printf("Error loading data from %s\n", argv[1]);
        return 1;
    }

    printf("Loaded %d points, K=%d\n", N, K);
    printf("Configuration: %d blocks, %d threads\n", blocks, threads);


    float *h_centroids = (float*)malloc(K * 2 * sizeof(float));
    if (!h_centroids) {
        printf("Error allocating memory for centroids\n");
        free(h_points);
        return 1;
    }

    // Asignar memoria en GPU
    float *d_points, *d_centroids;
    cudaMalloc(&d_points, N * 2 * sizeof(float));
    cudaMalloc(&d_centroids, K * 2 * sizeof(float));

    // Copiar puntos a GPU
    cudaMemcpy(d_points, h_points, N * 2 * sizeof(float), cudaMemcpyHostToDevice);

    

    initialize_random_centroids(h_points, N, h_centroids, K);

    printf("Initial centroids:\n");
    for (int c = 0; c < K; c++) {
        printf("C%d = (%.4f, %.4f)\n", c, h_centroids[2 * c], h_centroids[2 * c + 1]);
    }


    cudaMemcpy(d_centroids, h_centroids, K * 2 * sizeof(float), cudaMemcpyHostToDevice);

    // Crear eventos para medir tiempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("\n=== Running K-means GPU Basic ===\n");

    // Medir tiempo de ejecución
    cudaEventRecord(start);
    
    // Ejecutar K-means
    kmeans_gpu_basic(d_points, d_centroids, N, K, max_iters, epsilon, blocks, threads);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calcular tiempo transcurrido
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copiar centroides finales de vuelta a CPU
    cudaMemcpy(h_centroids, d_centroids, K * 2 * sizeof(float), cudaMemcpyDeviceToHost);

    // Imprimir resultados
    print_results(h_centroids, K, milliseconds);

    // Limpiar memoria
    cudaFree(d_points);
    cudaFree(d_centroids);
    free(h_points);
    free(h_centroids);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\nDone!\n");
    return 0;
}

// Implementación de las funciones del kernel
__device__ float dist2(float px, float py, float cx, float cy) {
    float dx = px - cx;
    float dy = py - cy;
    return dx * dx + dy * dy;
}

__global__ void assign_clusters_basic(
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

    for (int c = 0; c < K; c++) {
        float cx = centroids[2 * c];
        float cy = centroids[2 * c + 1];
        float d = dist2(px, py, cx, cy);
        if (d < best_dist) {
            best_dist = d;
            best_c = c;
        }
    }

    labels[idx] = best_c;
}

__global__ void update_centroids_basic(
    const float *points,
    float *centroids,
    const int *labels,
    float *sums,
    int *counts,
    int N, int K) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int c = labels[idx];
    float x = points[2 * idx];
    float y = points[2 * idx + 1];

    atomicAdd(&sums[2 * c], x);
    atomicAdd(&sums[2 * c + 1], y);
    atomicAdd(&counts[c], 1);
}

void kmeans_gpu_basic(
    float *d_points,
    float *d_centroids,
    int N, int K,
    int max_iters,
    float epsilon,
    int blocks,
    int threads) {
    
    int *d_labels;
    float *d_sums;
    int *d_counts;
    
    cudaMalloc(&d_labels, N * sizeof(int));
    cudaMalloc(&d_sums, K * 2 * sizeof(float));
    cudaMalloc(&d_counts, K * sizeof(int));

    for (int it = 0; it < max_iters; it++) {
        // Reset sums and counts
        cudaMemset(d_sums, 0, K * 2 * sizeof(float));
        cudaMemset(d_counts, 0, K * sizeof(int));

        // Assign clusters
        assign_clusters_basic<<<blocks, threads>>>(d_points, d_centroids, d_labels, N, K);
        cudaDeviceSynchronize();

        // Update centroids
        update_centroids_basic<<<blocks, threads>>>(d_points, d_centroids, d_labels, d_sums, d_counts, N, K);
        cudaDeviceSynchronize();

        // Compute new centroids and movement
        float movement = 0.0f;
        for (int c = 0; c < K; c++) {
            float old_centroid[2];
            
            cudaMemcpy(old_centroid, &d_centroids[2 * c], 2 * sizeof(float), cudaMemcpyDeviceToHost);
            
            int count;
            
            cudaMemcpy(&count, &d_counts[c], sizeof(int), cudaMemcpyDeviceToHost);
            
            float new_centroid[2] = {old_centroid[0], old_centroid[1]};
            if (count > 0) {
                float sum[2];
                
                cudaMemcpy(sum, &d_sums[2 * c], 2 * sizeof(float), cudaMemcpyDeviceToHost);
                new_centroid[0] = sum[0] / count;
                new_centroid[1] = sum[1] / count;
                
                
                cudaMemcpy(&d_centroids[2 * c], new_centroid, 2 * sizeof(float), cudaMemcpyHostToDevice);
            }

            float dx = new_centroid[0] - old_centroid[0];
            float dy = new_centroid[1] - old_centroid[1];
            movement += dx * dx + dy * dy;
        }

        printf("Iteration %d - centroid movement = %.6f\n", it, movement);
        if (movement < epsilon) {
            printf("Converged after %d iterations.\n", it);
            break;
        }
    }

    cudaFree(d_labels);
    cudaFree(d_sums);
    cudaFree(d_counts);
}