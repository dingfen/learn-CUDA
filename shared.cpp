#include <stdio.h>

__global__ void staticReverse(int *d, int n)
{
    __shared__ int s[64];
    int t = threadIdx.x;
    int tr = n-t-1;
    s[t] = d[t];
    __syncthreads();
    d[t] = s[tr];
}

__global__ void dynamicReverse(int *d, int n)
{
    extern __shared__ int s[];
    int t = threadIdx.x;
    int tr = n-t-1;
    s[t] = d[t];
    __syncthreads();
    d[t] = s[tr];
}

int main(void)
{
    const int n = 64;
    int a[n], r[n], d[n];

    for (int i = 0; i < n; i++) {
        a[i] = i;
        r[i] = n-i-1;
        d[i] = 0;
    }

    int *d_d;
    cudaMalloc(&d_d, n * sizeof(int));

    // run version with static shared memory
    cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
    //your code here
    staticReverse<<<1,n>>>(d_d, n);
    //end of your code
    cudaMemcpy(d, d_d, n*sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++)
        if (d[i] != r[i]) {
            printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);
            exit(-1);
        }
    printf("static success\n");

    // run dynamic shared memory version
    cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
    //your code here
    dynamicReverse<<<1,n,n*sizeof(int)>>>(d_d, n);
    //end of your code
    cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        if (d[i] != r[i]) {
            printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);
            exit(-1);
        }
    printf("dynamic success\n");
    }
}