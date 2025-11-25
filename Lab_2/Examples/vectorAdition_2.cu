#include <iostream>

void add(int n, float *x, float *y)
{
    for (int i = 0; i < n; i++)
    {
        y[i] = x[i] + y[i];
        
    }

    
}

int main(void)
{
    int N = 1 << 20; // 1M elements
    float *x = new float[N];
    float *y = new float[N];

    int size = N*sizeof(float);

    float *d_x, *d_y; // Copias de x y 
    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_y, size);

    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

    add<<1, 1>>(N, d_x, d_y);

    
    delete [] x; delete [] y;
    return 0;
}