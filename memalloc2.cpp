#include <iostream>
#include <cuda_runtime.h>
using namespace std;   
  
int main()  
{  
    float * pDeviceData = NULL;  
    int width = 10 * sizeof(float);  
    int height = 10 * sizeof(float);  
    size_t pitch;  
  
    cudaError err = cudaSuccess;  
  
    err = cudaMallocPitch(&pDeviceData, &pitch, width, height);     
    if (err != cudaSuccess)  
    {  
        cout << "call cudaMallocPitch fail!!!" << endl;  
        exit(1);  
    }  
    cout << "width: " << width << endl;  
    cout << "height: " << height << endl;  
    cout << "pitch: " << pitch << endl;  
  
    cudaFree(pDeviceData);  
    return 0;  
}  