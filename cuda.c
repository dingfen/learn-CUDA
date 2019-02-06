#include <stdio.h>

__global__ void myKernel()
{

}

int main(int argc, char const *argv[])
{
    myKernel<<<4,2>>>();
    printf("Hello world\n");
    return 0;
}
