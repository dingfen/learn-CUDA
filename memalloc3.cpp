#include <iostream>
#include <cuda_runtime.h>  
using namespace std;  
  
int main()  
{  
    cudaError err = cudaSuccess;  
  
    cudaPitchedPtr pitchPtr;  
    cudaExtent extent;  
    extent.width = 10 * sizeof(float);  
    extent.height = 22 * sizeof(float);  
    extent.depth = 33 * sizeof(float);  
  
    err = cudaMalloc3D(&pitchPtr, extent);  
    if (err != cudaSuccess)  
    {  
        cout << "call cudaMalloc3D fail!!!" << endl;  
        exit(1);  
    }  
    cout << "width: " << extent.width << endl;          
    cout << "height: " << extent.height << endl;  
    cout << "depth: " << extent.depth << endl;  
  
    cout << endl;  
    cout << "pitch: " << pitchPtr.pitch << endl;      
    cout << "xsize: " << pitchPtr.xsize << endl;     
    cout << "ysize: " << pitchPtr.ysize << endl;     
  
    cudaFree(pitchPtr.ptr);   
    return 0;  
}  