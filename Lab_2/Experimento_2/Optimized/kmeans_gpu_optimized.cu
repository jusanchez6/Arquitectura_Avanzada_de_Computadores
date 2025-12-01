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

#define MAX_POINTS 100000
#define MAX_CLUSTERS 20
#define THREADS_PER_BLOCK 256

// ==========================================
// GPU distance
// ==========================================
__device__ float dist2(float px, float py, float cx, float cy) {
    float dx = px - cx;
    float dy = py - cy;
    return dx*dx + dy*dy;
}

// ==========================================
// ASSIGN CLUSTERS (optimized)
// ==========================================
__global__ void assign_clusters_advanced(
    const float *points,
    const float *centroids,
    int *labels,
    int N, int K)
{
    __shared__ float shared_centroids[MAX_CLUSTERS * 2];

    // Load centroids into shared memory
    for (int i = threadIdx.x; i < K * 2; i += blockDim.x) {
        shared_centroids[i] = centroids[i];
    }
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

// ==========================================
// UPDATE CENTROIDS (optimized)
// ==========================================
__global__ void update_centroids_advanced(
    const float *points,
    float *centroids,
    const int *labels,
    int N, int K)
{
    extern __shared__ float shared_data[];
    float *sums = shared_data;
    int *counts = (int*)&sums[K * 2];

    for (int i = threadIdx.x; i < K * 2; i += blockDim.x)
        sums[i] = 0.0f;

    for (int i = threadIdx.x; i < K; i += blockDim.x)
        counts[i] = 0;

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

    if (threadIdx.x < K && counts[threadIdx.x] > 0) {
        centroids[2 * threadIdx.x]     = sums[2 * threadIdx.x] / counts[threadIdx.x];
        centroids[2 * threadIdx.x + 1] = sums[2 * threadIdx.x + 1] / counts[threadIdx.x];
    }
}

// ==========================================
// CREATE OUTPUT DIRECTORY
// ==========================================
void ensure_output_dir() {
    struct stat st = {0};
    if (stat("output_optimized", &st) == -1) {
        mkdir("output_optimized", 0777);
    }
}

// ==========================================
// SAVE LABELS
// ==========================================
void save_labels(const int *h_labels, int N, int iter) {
    char filename[128];
    sprintf(filename, "labels_iter_%d.csv", iter);

    FILE *f = fopen(filename, "w");
    for (int i = 0; i < N; i++)
        fprintf(f, "%d\n", h_labels[i]);
    fclose(f);
}

// ==========================================
// SAVE CENTROIDS
// ==========================================
void save_centroids(const float *h_centroids, int K, int iter) {
    char filename[128];
    sprintf(filename, "centroids_iter_%d.csv", iter);

    FILE *f = fopen(filename, "w");
    for (int c = 0; c < K; c++)
        fprintf(f, "%f,%f\n", h_centroids[2*c], h_centroids[2*c+1]);
    fclose(f);
}

// ==========================================
// K-MEANS OPTIMIZED (with saving)
// ==========================================
void kmeans_advanced(
    float *d_points,
    float *d_centroids,
    int N,
    int K,
    int max_iters,
    float epsilon,
    int blocks,
    int threads)
{
    int *d_labels;
    cudaMalloc(&d_labels, N * sizeof(int));

    int *h_labels = (int*)malloc(N * sizeof(int));
    float *h_centroids = (float*)malloc(K * 2 * sizeof(float));

    size_t shared_mem = (K * 2 * sizeof(float)) + (K * sizeof(int));

    for (int it = 0; it < max_iters; it++) {

        assign_clusters_advanced<<<blocks, threads>>>(d_points, d_centroids, d_labels, N, K);
        cudaDeviceSynchronize();

        cudaMemcpy(h_labels, d_labels, N * sizeof(int), cudaMemcpyDeviceToHost);
        save_labels(h_labels, N, it);

        update_centroids_advanced<<<blocks, threads, shared_mem>>>(d_points, d_centroids, d_labels, N, K);
        cudaDeviceSynchronize();

        cudaMemcpy(h_centroids, d_centroids, K * 2 * sizeof(float), cudaMemcpyDeviceToHost);
        save_centroids(h_centroids, K, it);

        printf("Iteration %d saved.\n", it);
    }

    free(h_labels);
    free(h_centroids);
    cudaFree(d_labels);
}

// ==========================================
// CSV LOADING + MAIN
// ==========================================
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

void initialize_centroids_simple(float *centroids, int K) {
    srand(time(NULL));
    for (int c = 0; c < K; c++) {
        centroids[2*c]   = (rand()%200 - 100) / 10.0f;
        centroids[2*c+1] = (rand()%200 - 100) / 10.0f;
    }
}

void print_results(float *centroids, int K, float ms) {
    printf("\nFinal centroids:\n");
    for (int c = 0; c < K; c++)
        printf("C%d = (%.4f, %.4f)\n", c, centroids[2*c], centroids[2*c+1]);
    printf("Execution time: %.2f ms\n", ms);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s data.csv [K] [blocks] [threads]\n", argv[0]);
        return 1;
    }

    int K = (argc > 2) ? atoi(argv[2]) : 3;
    int blocks = (argc > 3) ? atoi(argv[3]) : 64;
    int threads = (argc > 4) ? atoi(argv[4]) : 256;
    int max_iters = 10;

    float *h_points = NULL;
    int N = load_csv(argv[1], &h_points);
    if (N <= 0) return 1;

    float *h_centroids = (float*)malloc(K * 2 * sizeof(float));
    initialize_centroids_simple(h_centroids, K);

    float *d_points, *d_centroids;
    cudaMalloc(&d_points, N * 2 * sizeof(float));
    cudaMalloc(&d_centroids, K * 2 * sizeof(float));

    cudaMemcpy(d_points, h_points, N * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids, K * 2 * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    kmeans_advanced(d_points, d_centroids, N, K, max_iters, 1e-6f, blocks, threads);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_centroids, d_centroids, K * 2 * sizeof(float), cudaMemcpyDeviceToHost);

    print_results(h_centroids, K, ms);

    free(h_centroids);
    free(h_points);
    cudaFree(d_points);
    cudaFree(d_centroids);

    printf("\nOptimized version complete.\n");
    return 0;
}
