#include <iostream>
#include <iterator>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstdlib>
using namespace std;

void checkResult(int *vec1, int *vec2, int *resultVec, int size) {
  for (int i = 0; i < size; i++) {
    assert(resultVec[i] == vec1[i] + vec2[i]);
  }
}




// CUDA kernel for vector addition
__global__ void addVectors(int* vec1, int* vec2, int* resultVec, int size) {
  int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (threadId < size) {
    resultVec[threadId] = vec1[threadId] + vec2[threadId];
  }
}



int main() {
  constexpr int arraySize = 1 << 26;
  size_t memorySize = sizeof(int) * arraySize;

  int *hostVec1, *hostVec2, *hostResult;
  cudaMallocHost(&hostVec1, memorySize);
  cudaMallocHost(&hostVec2, memorySize);
  cudaMallocHost(&hostResult, memorySize);

  
  for(int i = 0; i < arraySize; i++){
    hostVec1[i] = rand() % 100;
    hostVec2[i] = rand() % 100;
  }



  
  int *devVec1, *devVec2, *devResult;
  cudaMalloc(&devVec1, memorySize);
  cudaMalloc(&devVec2, memorySize);
  cudaMalloc(&devResult, memorySize);

  cudaMemcpy(devVec1, hostVec1, memorySize, cudaMemcpyHostToDevice);
  cudaMemcpy(devVec2, hostVec2, memorySize, cudaMemcpyHostToDevice);



  
  int threadsPerBlock = 1 << 10;
  int blocksPerGrid = (arraySize + threadsPerBlock - 1) / threadsPerBlock;

  
  addVectors<<<blocksPerGrid, threadsPerBlock>>>(devVec1, devVec2, devResult, arraySize);
  cudaMemcpy(hostResult, devResult, memorySize, cudaMemcpyDeviceToHost);

  
  checkResult(hostVec1, hostVec2, hostResult, arraySize);


  
  cudaFreeHost(hostVec1);
  cudaFreeHost(hostVec2);
  cudaFreeHost(hostResult);
  cudaFree(devVec1);
  cudaFree(devVec2);
  cudaFree(devResult);


  
  cout << "COMPLETED SUCCESSFULLY\n";

  return 0;
}
