#include <iostream>
using namespace std;

__global__ void CudaAddition(int* x, int* y, int* z, int N){
      int index = blockIdx.x * blockDim.x + threadIdx.x;

      if (index < N) {
          z[index] = x[index]+y[index];
      }
}

int main(){
    int N = 1e6;

    // int *x = (int *)malloc(sizeof(int)*N);
    int *x = new int[N];
    int *y = new int[N];
    int *z = new int[N];
    // int *y = (int *)malloc(sizeof(int)*N);
    // int *z = (int *)malloc(sizeof(int)*N);
    cout<<"started"<<endl;

    for(int i=0;i<N;i++){
          x[i]=2;
          y[i]=2;
    }


    int *a , *b , *c;
    cudaMalloc(&a , sizeof(int)*N);
    cudaMalloc(&b , sizeof(int)*N);
    cudaMalloc(&c , sizeof(int)*N);

    cudaMemcpy(a, x, sizeof(int)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(b, y, sizeof(int)*N, cudaMemcpyHostToDevice);

    dim3 th(N);
    CudaAddition<<<1, th>>>(a, b, c, N);

    cudaMemcpy(z, c, sizeof(int)*N, cudaMemcpyDeviceToHost);

    // for(int i=0;i<N;i++){
    //     cout<<z[i]<<" ";
    // }

    cout<<"completed"<<endl;

    return 0;
}