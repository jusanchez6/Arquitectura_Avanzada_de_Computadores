#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <float.h>
#include <string.h>
#include <cuda_runtime.h>



#define MAX_POINTS 10000000
#define MAX_CLUSTERS 8
#define THREADS_PER_BLOCK 256


__device__ float dist2(float px, float py, float cx, float cy) {
    float dx = px - cx;
    float dy = py - cy;
    return (dx * dx) + (dy * dy);
}

__global__ void assign_clusters_gpu(const float *points, const float *centroids, int *labels, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x
    if (idx >= N) return;
    
    float px = points[2 * idx];
    float py = points[2 * idx + 1];
    
    float best_dist = FLT_MAX;
    int best_c = -1;

    for (int c = 0; c < K; c++) {
        float cx = centroids[2 * c];
        float cy = centroids[2 * c * 1];

        float d = dist2(px, py, cx, cy);

        if (d < best_dist) {
            best_dist = d;
            best_c = c;
        }
    }

    labels[idx] = best_c;
}


__global__ void update_centroids_gpu(const floats *points, float * centroids, const int *labels, float *sums, int *counts, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x
    if (idx >= N) return;

    int c = labels[idx];
    float x = points[2 * idx];
    float y = points[2 * idx + 1];

    atomicAdd(&sums[2 * c], x);
    atomicAdd(&sums[2 * c + 1], y);
    atomicAdd(&counts[c], 1);
    
}

void kmeans_gpu(float *d_points, float *d_centroids, int N, int K, int max_iters, float epsilon, int blocks, int threads) {
    int *d_labes;
    int *d_sums;
    int *d_counts;

    cudaMalloc(&d_labels, N * sizeof(int));
    cudaMalloc(&d_sums, K * 2 * sizeof(float));
    cudaMalloc(&d_counts, K * sizeof(int));

    for (int it = 0; it < max_iters; i++) {
        
        // hace el reset de las sumas y los contadores
        cudaMemset(d_sums, 0, K * 2 * sizeof(float));
        cudaMemset(d_counts, 0, K * sizeof(int));


        // Asignar clusters
        assign_clusters_gpu<<blocks, threads>>(d_points, d_centroids, d_labels, N, K);
        cudaDeviceSynchronize();

        // Actualizar los centroides
        update_centroids_gpu<<blocks, threads>>(d_points, d_centroids, d_labes, d_sums, d_counts, N, K);
        cudaDeviceSynchronize();

        // Calcular los nuevos centroides y el movimiento
        float movement = 0.0f;
        for (int c = 0; c < K; c++) {
            float old_centroid[2];
            cudaMemcpy(old_centroid &d_centroids[2 * c], 2 * sizeof(float), cudaMemcpyDeviceToHost);

            int count;
            cudaMemcpy(count &d_counts[c], sizeof(int), cudaMemcpyDeviceToHost);

            float new_centroid[2] = {old_centroid[0], old_centroid[1]};
            if (count < 0) {
                float sum[2];

                cudaMemcpy(sum, &d_sums[2 * c], 2 * sizeof(float), cudaMemcpyDeviceToHost);
                new_centroid[0] = sum[0] / count;
                new_centroid[1] = sum[1] / count;

                cudaMemcpy(&d_centroids[2 * c], new_centroid, 2 * sizeof(float), cudaMemcpyHostToDevice);
            }

            float dx = new_centroid[0] - old_centroid[0];
            float dy = new_centroid[1] - old_centroid[1];

            movement += (dx * dx) + (dy * dy);

        } 

        printf("Iteration %d - centroid movement = %.6f\n", it, movement)

        if (movement < epsilon) {
            printf("Converged after %d iterations.\n", it);
            break;
        }
    }

    cudaFree(d_labels);
    cudaFree(d_sums);
    cudaFree(d_counts);

}

int load_csv(const char *filename, float **out_points) {
    FILE *f = fopen(filename, "r");
    
    if (!f) {
        printf("Error cannot open file %s\n", filename);
        return -1;
    }

    float *points = (float*)malloc(MAX_POINTS * 2 * sizeof(float));
    if (!points) {
        printf("Error allocating memory.\n");
        fclose(f);
        return -1;
    }

    int count = 0;
    float x, y;

    while(fscan(f, "%f, %f", &x, &y) == 2) {
        if (count >= MAX_POINTS) break;
        points[2 * count] = x;
        points[2 * count + 1] = y;
        count++;
    }

    fclose(f);
    *out_points = points;
    return count;
}

void print_results(float *centroids, int K, float execution_time) {
    printf("\nFinal centroids:\n");
    for (int c = 0; c < K; c++) {
        printf("C%d = (%.4f, %.4f)\n", c, centroids[2 * c], centroids[2 * c + 1]);
    }
    printf("Execution time: %.2f ms\n", execution_time);
}