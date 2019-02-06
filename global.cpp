#include <cuda_runtime.h>
#include <stdio.h>

__device__ float devData;
__global__ void checkGlobalVariable() 
{
    // display the original value
    printf("Device: the value of the global variable is %f\n",devData);
    // alter the value
    devData +=2.0f;
}

int main(void) 
{
    float value = 3.14f;
	//your code here
    cudaMemcpyToSymbol(devData, &value, sizeof(float));
	//end of your code
    printf("Host: copied %f to the global variable\n", value);
    checkGlobalVariable <<<1, 1>>>();
    cudaMemcpyFromSymbol(&value, devData, sizeof(float));
    printf("Host: the value changed by the kernel to %f\n", value);
    cudaDeviceReset();
    return EXIT_SUCCESS;
}    