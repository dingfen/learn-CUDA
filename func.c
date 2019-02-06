#include <stdio.h>

__device__ int dev1() {
}

__device__ int dev2() {
}

__global__ void run10Times() {
	//your code here
	dev1();
	dev2();
	//end of your code
}

int main() {
	run10Times<<<2, 5>>>();
	printf("Hello, World!\n");
	return 0;
}