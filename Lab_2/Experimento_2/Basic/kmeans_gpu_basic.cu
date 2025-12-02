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


/**
 * @brief Calcula la distancia euclidiana al cuadrado entre dos puntos 2D.
 *
 * Esta función devuelve (px - cx)^2 + (py - cy)^2, evitando la operación de
 * raíz cuadrada, lo cual es útil para comparaciones de distancias.
 *
 * @param px Coordenada X del punto.
 * @param py Coordenada Y del punto.
 * @param cx Coordenada X del centro.
 * @param cy Coordenada Y del centro.
 * @return float Distancia al cuadrado entre (px, py) y (cx, cy).
 */
__device__ float dist2(float px, float py, float cx, float cy) {
    float dx = px - cx;
    float dy = py - cy;
    return dx * dx + dy * dy;
}

/**
 * @brief Asigna cada punto al clúster más cercano.
 *
 * Kernel que calcula para cada punto 2D el centroide más cercano usando
 * distancia euclidiana al cuadrado. Cada hilo procesa un punto y escribe
 * la etiqueta correspondiente.
 *
 * @param points Arreglo de puntos 2D en formato [x0, y0, x1, y1, ...].
 * @param centroids Arreglo de centroides 2D en formato [cx0, cy0, cx1, cy1, ...].
 * @param labels Arreglo de salida donde se guarda para cada punto el índice del clúster asignado.
 * @param N Número total de puntos.
 * @param K Número total de centroides.
 * @return void No retorna valor; escribe los resultados en `labels`.
 */
__global__ void assign_clusters_basic(
    const float *points,
    const float *centroids,
    int *labels,
    int N, int K)
{
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

/**
 * @brief Acumula las sumas parciales para recalcular los centroides.
 *
 * Kernel que, para cada punto, añade sus coordenadas a la suma acumulada
 * de su clúster correspondiente y aumenta el contador de puntos.  
 * Estos valores luego se usan para calcular los nuevos centroides.
 *
 * @param points Arreglo de puntos 2D en formato [x0, y0, x1, y1, ...].
 * @param centroids (No usado en esta versión básica, pero se mantiene por consistencia).
 * @param labels Etiquetas que indican el clúster asignado a cada punto.
 * @param sums Arreglo donde se acumulan las sumas de coordenadas por clúster.
 * @param counts Arreglo donde se cuenta cuántos puntos pertenecen a cada clúster.
 * @param N Número total de puntos.
 * @param K Número total de clústeres.
 * @return void No retorna valor; actualiza `sums` y `counts`.
 */
__global__ void update_centroids_basic(
    const float *points,
    float *centroids,

    const int *labels,
    float *sums,
    int *counts,
    int N, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int c = labels[idx];
    float x = points[2 * idx];
    float y = points[2 * idx + 1];

    atomicAdd(&sums[2 * c], x);
    atomicAdd(&sums[2 * c + 1], y);
    atomicAdd(&counts[c], 1);
}

/**
 * @brief Carga puntos 2D desde un archivo CSV.
 *
 * Lee un archivo CSV con formato "x,y" por línea y almacena los valores
 * en un arreglo dinámico de floats organizado como [x0, y0, x1, y1, ...].
 * El arreglo es asignado internamente y devuelto mediante `out_points`.
 *
 * @param filename Nombre del archivo CSV a leer.
 * @param out_points Puntero de salida donde se almacenará la dirección del arreglo dinámico.
 * @return int Número de puntos cargados, o -1 en caso de error.
 */
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

/**
 * @brief Imprime los centroides finales y el tiempo total de ejecución.
 *
 * Muestra por consola las coordenadas de cada centroide calculado y el
 * tiempo total que tomó el algoritmo en milisegundos.
 *
 * @param centroids Arreglo de centroides en formato [cx0, cy0, cx1, cy1, ...].
 * @param K Número de centroides.
 * @param execution_time Tiempo total de ejecución del algoritmo (en ms).
 * @return void No retorna valor.
 */
void print_results(float *centroids, int K, float execution_time) {
    printf("\nFinal centroids:\n");
    for (int c = 0; c < K; c++) {
        printf("C%d = (%.4f, %.4f)\n", c, centroids[2 * c], centroids[2 * c + 1]);
    }
    printf("Execution time: %.2f ms\n", execution_time);
}

/**
 * @brief Inicializa los centroides escogiendo puntos aleatorios del conjunto.
 *
 * Selecciona aleatoriamente K puntos del arreglo de entrada y usa sus
 * coordenadas como los centroides iniciales para el algoritmo k-means.
 *
 * @param points Arreglo de puntos 2D en formato [x0, y0, x1, y1, ...].
 * @param N Número total de puntos disponibles.
 * @param centroids Arreglo donde se almacenarán los centroides iniciales.
 * @param K Número de centroides a inicializar.
 * @return void No retorna valor.
 */
void initialize_random_centroids(float *points, int N, float *centroids, int K) {
    srand(time(NULL));
    for (int c = 0; c < K; c++) {
        int idx = rand() % N;
        centroids[2 * c] = points[2 * idx];
        centroids[2 * c + 1] = points[2 * idx + 1];
    }
}

/**
 * @brief Ejecuta la versión básica del algoritmo k-means en GPU.
 *
 * Implementa k-means usando CUDA: asignación de clústeres, acumulación de
 * centroides, actualización de posiciones y verificación de convergencia.
 * Además, guarda en archivos CSV las etiquetas y centroides en cada iteración
 * para facilitar la visualización del proceso.
 *
 * @param d_points Puntero en GPU a los puntos 2D ([x0,y0,x1,y1,...]).
 * @param d_centroids Puntero en GPU a los centroides actuales.
 * @param N Número total de puntos.
 * @param K Número de clústeres.
 * @param max_iters Iteraciones máximas permitidas.
 * @param epsilon Umbral de convergencia basado en el movimiento total de centroides.
 * @param blocks Número de bloques para los kernels.
 * @param threads Número de hilos por bloque.
 * @return void No retorna valor; actualiza los centroides directamente en GPU.
 */
void kmeans_gpu_basic(
    float *d_points,
    float *d_centroids,
    int N, int K,
    int max_iters,
    float epsilon,
    int blocks,
    int threads)
{
    int *d_labels;
    float *d_sums;
    int *d_counts;

    cudaMalloc(&d_labels, N * sizeof(int));
    cudaMalloc(&d_sums, K * 2 * sizeof(float));
    cudaMalloc(&d_counts, K * sizeof(int));

    for (int it = 0; it < max_iters; it++) {

        // Reset
        cudaMemset(d_sums, 0, K * 2 * sizeof(float));
        cudaMemset(d_counts, 0, K * sizeof(int));

        // Assign clusters
        assign_clusters_basic<<<blocks, threads>>>(d_points, d_centroids, d_labels, N, K);
        cudaDeviceSynchronize();

        // Save labels for visualization
        int *h_labels = (int*)malloc(N * sizeof(int));
        cudaMemcpy(h_labels, d_labels, N * sizeof(int), cudaMemcpyDeviceToHost);

        char file_labels[64];
        sprintf(file_labels, "labels_iter_%d.csv", it);
        FILE *fl = fopen(file_labels, "w");
        for (int i = 0; i < N; i++) fprintf(fl, "%d\n", h_labels[i]);
        fclose(fl);
        free(h_labels);

        // Update centroids
        update_centroids_basic<<<blocks, threads>>>(
            d_points, d_centroids, d_labels, d_sums, d_counts, N, K
        );
        cudaDeviceSynchronize();

        float movement = 0.0f;

        // Host storage
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

        // Save centroids after this iteration
        char file_centroids[64];
        sprintf(file_centroids, "centroids_iter_%d.csv", it);
        FILE *fc = fopen(file_centroids, "w");

        float h_centroids[MAX_CLUSTERS * 2];
        cudaMemcpy(h_centroids, d_centroids, K * 2 * sizeof(float), cudaMemcpyDeviceToHost);

        for (int c = 0; c < K; c++) {
            fprintf(fc, "%f,%f\n", h_centroids[2 * c], h_centroids[2 * c + 1]);
        }
        fclose(fc);

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

/**
 * @brief Punto de entrada del programa para ejecutar k-means en GPU.
 *
 * Carga un archivo CSV con puntos 2D, inicializa los centroides,
 * transfiere los datos a la GPU y ejecuta la versión básica del
 * algoritmo k-means. Mide el tiempo total de ejecución y muestra
 * los centroides finales.  
 *
 * Parámetros de ejecución (opcionales):
 * - K: número de clústeres (por defecto 3)
 * - blocks: cantidad de bloques CUDA (por defecto 64)
 * - threads: hilos por bloque (por defecto 256)
 *
 * @param argc Número de argumentos de línea de comandos.
 * @param argv Lista de argumentos: data.csv [K] [blocks] [threads].
 * @return int 0 si la ejecución fue exitosa, o 1 si ocurrió un error.
 */
int main(int argc, char **argv) {

    if (argc < 2) {
        printf("Usage: %s data.csv [K] [blocks] [threads]\n", argv[0]);
        return 1;
    }

    int K = (argc > 2) ? atoi(argv[2]) : 3;
    int blocks = (argc > 3) ? atoi(argv[3]) : 64;
    int threads = (argc > 4) ? atoi(argv[4]) : 256;
    int max_iters = 20;
    float epsilon = 1e-4f;

    float *h_points = NULL;
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

    // Timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Run K-means
    kmeans_gpu_basic(d_points, d_centroids, N, K, max_iters, epsilon, blocks, threads);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_centroids, d_centroids, K * 2 * sizeof(float), cudaMemcpyDeviceToHost);

    print_results(h_centroids, K, ms);

    cudaFree(d_points);
    cudaFree(d_centroids);
    free(h_points);
    free(h_centroids);

    return 0;
}