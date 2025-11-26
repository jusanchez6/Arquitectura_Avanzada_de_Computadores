% % writefile dot_product.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include <ctime>

#define TPB 1024 // Threads per block

    unsigned t0,
    t1;

/**
 * @brief Kernel que reduce (suma) un vector en segmentos: cada bloque suma un subconjunto y escribe un parcial.
 *
 * reduce_stage toma un vector v[0..length-1] y, para cada bloque, copia a memoria compartida los elementos
 * correspondientes (si el gid del hilo está fuera de bounds guarda 0) y hace la reducción tradicional (halving).
 * El hilo 0 escribe el parcial en v[blockIdx.x].
 *
 * @param v Vector en device que contiene los valores a reducir; tras la ejecución los primeros gridDim.x
 *          elementos contienen las sumas parciales por bloque.
 * @param length Longitud válida de v en esta llamada (número de elementos que contienen datos útiles).
 */
__global__ void reduce_stage(float *v, int length)
{
    __shared__ float buf[TPB];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Copiar del vector global
    if (gid < length)
        buf[threadIdx.x] = v[gid];
    else
        buf[threadIdx.x] = 0.0f;

    __syncthreads();

    // Reducción
    for (int step = blockDim.x / 2; step > 0; step >>= 1)
    { // que tantos hilos hay en ese bloque?
        if (threadIdx.x < step)
        { // pues los vamos a sumar con pasos de la mitad de los hilos en cada pasada
            buf[threadIdx.x] += buf[threadIdx.x + step];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        v[blockIdx.x] = buf[0];
}

/**
 * @brief Kernel que calcula productos element-wise y realiza la reducción local por bloque.
 *
 * Cada hilo acumula x[i]*y[i] para índices i = index, index + stride, ...
 * Luego se realiza una reducción en memoria compartida para obtener una suma parcial por bloque,
 * que se escribe en out[blockIdx.x].
 *
 * @param n Tamaño del vector (n elementos).
 * @param a Puntero a vector a (device).
 * @param b Puntero a vector b (device).
 * @param out Vector de salida (device) de longitud >= gridDim.x; out[blockIdx.x] = suma parcial del bloque.
 */
__global__ void dot_kernel(int n, const float *a, const float *b, float *out)
{
    __shared__ float local[TPB];

    local[threadIdx.x] = 0.0f;
    __syncthreads();

    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = start; i < n; i += stride)
    {
        local[threadIdx.x] += a[i] * b[i];
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            local[threadIdx.x] += local[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        out[blockIdx.x] = local[0];
}

void do_dot(int total_blocks, int block_size)
{
    t0 = clock();

    const int N = 1 << 28; // 268M floats

    int remaining_blocks;

    float *h_x = new float[N];
    float *h_y = new float[N];

    float *d_x, *d_y, *d_partial;
    float result = 0.0f;

    // Inicializar
    for (int i = 0; i < N; i++)
    {
        h_x[i] = 3.0f;
        h_y[i] = 2.0f;
    }

    // Reservas GPU
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_partial, total_blocks * sizeof(float));

    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Primera reducción
    dot_kernel<<<total_blocks, block_size>>>(N, d_x, d_y, d_partial);
    cudaDeviceSynchronize();

    // Segunda reducción
    remaining_blocks = (total_blocks + block_size - 1) / block_size;

    reduce_stage<<<remaining_blocks, block_size>>>(d_partial, total_blocks);
    cudaDeviceSynchronize();

    while (remaining_blocks > 1)
    {
        int prev = remaining_blocks;
        remaining_blocks = (remaining_blocks + block_size - 1) / block_size;

        reduce_stage<<<remaining_blocks, block_size>>>(d_partial, prev);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(&result, d_partial, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Dot product result: %.2f\n", result);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_partial);
    delete[] h_x;
    delete[] h_y;

    t1 = clock();

    double time = (double(t1 - t0) / CLOCKS_PER_SEC);

    printf("Excution time: %2f\n", time);
}

int main()
{

    const int block_size = TPB;
    int sms;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);

    
    for (int tb = 1 << 6; tb < 1 << 9; tb <<= 1)
    {
        printf("Numero de bloques: %d", total_blocks);
        do_dot(tb, block_size)
    }

    printf("\n\n\n");

    int total_blocks = sms * 32;

    for (int bs = 1 << 6; bs < 1 << 9; bs <<= 1)
    {
        printf("Numero de bloques: %d", total_blocks);
        do_dot(total_blocks, bs)
    }

    return 0
}