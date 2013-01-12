#include    <wb.h>

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

const int TILE_WIDTH = 16;

__global__ void matrixMultiplyShared(float * A, float * B, float * C,
			             int numARows, int numAColumns,
			             int numBRows, int numBColumns,
			             int numCRows, int numCColumns) {

  __shared__ float As[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x; 
  int by = blockIdx.y;
  
  int tx = threadIdx.x; 
  int ty = threadIdx.y;
  
  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;
  
  float val = 0;

  for( int i = 0; i != numAColumns/TILE_WIDTH; ++i ) {
    if (row >= numARows || (i*TILE_WIDTH + tx) >= numBColumns) {
      As[ty][tx] = 0;
      Bs[ty][tx] = 0;
    } 
    else {
      As[ty][tx] = A[row * numAColumns + (i * TILE_WIDTH + tx)];
      Bs[ty][tx] = B[(i * TILE_WIDTH + ty) * numBColumns + col];      
    }
    __syncthreads();

    for( int k = 0; k != TILE_WIDTH; ++k )
      val += As[ty][k] * Bs[k][tx];

    __syncthreads();
  }

  C[row * numCColumns + col] = val;
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows; // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);

    numCRows = numARows;
    numCColumns = numBColumns;
    hostC = (float *) malloc(numCRows * numCColumns * sizeof(float));

    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
  	wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    int a_size = numAColumns * numARows * sizeof(float);
    wbCheck(cudaMalloc((void **)&deviceA, a_size));
    int b_size = numBColumns * numBRows * sizeof(float);
    wbCheck(cudaMalloc((void **)&deviceB, b_size));
    int c_size = numCColumns * numCRows * sizeof(float);
    wbCheck(cudaMalloc((void **)&deviceC, c_size));

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy( deviceA, hostA, a_size, cudaMemcpyHostToDevice ));
    wbCheck(cudaMemcpy( deviceB, hostB, b_size, cudaMemcpyHostToDevice ));

    wbTime_stop(GPU, "Copying input memory to the GPU.");

    dim3 block_dim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 grid_dim((numBColumns + block_dim.x - 1) / block_dim.x,
                  (numARows + block_dim.y - 1) / block_dim.y, 1);
 
  
  	wbLog(TRACE, "grid_dim.x = ", grid_dim.x, " grid_dim.y = ", grid_dim.y);
  	wbLog(TRACE, "block_dim.x = ", block_dim.x, " block_dim.y = ", block_dim.y);
    
    wbTime_start(Compute, "Performing CUDA computation");
    matrixMultiplyShared<<<grid_dim, block_dim>>>( deviceA, deviceB, deviceC, 
                                                   numARows, numAColumns,
                                                   numBRows, numBColumns,
                                                   numCRows, numCColumns );

    cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy( hostC, deviceC, c_size, cudaMemcpyDeviceToHost ));
    wbTime_stop(Copy, "Copying output memory to the CPU");
  

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree( deviceA );
    cudaFree( deviceB );
    cudaFree( deviceC );

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}
