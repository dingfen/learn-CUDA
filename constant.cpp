#include<cuda_runtime.h>  
#include<iostream>  
using namespace std;  
  
__constant__ float num[40];  
__global__ void exchangeKernel(float *a)  
{  
    int offset = threadIdx.x + blockDim.x * blockIdx.x;  
    a[offset] = num[offset];  
}  
  
int main(){  
    float *devA,tmp[40],res[40];  
    cudaMalloc((void**)&devA, 40*sizeof(float));  
    for (int i = 0; i < 40; i++)
        tmp[i] = i*15;  
	//your code here
    cudaMemcpyToSymbol(num, tmp, 40 * sizeof(float));  
	//end of your code
    exchangeKernel<<<4, 10 >>>(devA);  
    cudaMemcpy(res, devA, 40 * sizeof(float), cudaMemcpyDeviceToHost);  
    for (int i = 0; i < 40; i++){  
        cout << res[i] << " " << endl;  
    }  
    return 0;  
}  