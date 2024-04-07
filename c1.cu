#include <iostream>
#include <cuda_runtime.h>
#include <cstring>
#include <ctime>
#include <cudnn.h>

const int C = 3;
const int H = 1024;
const int W = 1024;
const int K = 64;
const int FH = 3;
const int FW = 3;
const int P = 1;

#define checkCUDNN(expression)                             \
{                                                          \
    cudnnStatus_t status = (expression);                  \
    if (status != CUDNN_STATUS_SUCCESS) {                  \
        std::cerr << "Error on line " << __LINE__ << ": " \
                  << cudnnGetErrorString(status) << std::endl; \
        std::exit(EXIT_FAILURE);                          \
    }                                                      \
}

double check_sum(double* h_O, int O_size){
    double checksum = 0;
    for (int i = 0; i < O_size; i++) {
        checksum += h_O[i];
    }

    return checksum;
}

void print_tensor(const char* tensor_name, double* tensor, int K, int C, int H, int W) {
    std::cout << tensor_name << std::endl;

    for (int k = 0; k < K; k++) {
        if (K > 1) {
            std::cout << "Filter " << k << ":" << std::endl;
        }
        for (int c = 0; c < C; c++) {
            std::cout << "Channel " << c << ":" << std::endl;
            for (int x = 0; x < H; x++) {
                for (int y = 0; y < W; y++) {
                    int index = k * (C * H * W) + c * (H * W) + x * W + y;
                    std::cout << tensor[index] << " ";
                }
                std::cout << std::endl;
            }
        }
    }
}

__global__ void convolution_kernel(double* d_I0, double* d_F, double* d_O) {
    int k = blockIdx.z;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < W && y < H) {
        double sum = 0;
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < FW; i++) {
                for (int j = 0; j < FH; j++) {
                    sum += d_F[k * C * FW * FH + c * FW * FH + (FW - 1 - i) * FW + (FH - 1 - j)] * 
                    d_I0[c * ((H + 2 * P) * (W + 2 * P)) + (x + i) * (W + 2 * P) + (y + j)];
                }
            }
        }
        d_O[(k * H * W) + x * H + y] = sum;
    }
}

__global__ void tiling_conv_kernel(double* d_I0, double* d_F, double* d_O){
    // int k = blockIdx.z;
    // int x = blockIdx.x * blockDim.x + threadIdx.x;
    // int y = blockIdx.y * blockDim.y + threadIdx.y;

    // __shared__ double input_tile[C][32 + FH - 1][32 + FW - 1]; 

    // // Load input tiles into shared memory
    // for (int c = 0; c < C; c++) {
    //     int input_x = x - P + threadIdx.x;
    //     int input_y = y - P + threadIdx.y;

    //     if (input_x >= 0 && input_x < (W + 2 * P) && input_y >= 0 && input_y < (H + 2 * P)) {
    //         input_tile[c][threadIdx.y][threadIdx.x]= d_I0[c * ((H + 2 * P) * (W + 2 * P)) + input_y * (W + 2 * P) + input_x];
    //     } else {
    //         input_tile[c][threadIdx.y][threadIdx.x] = 0;
    //     }
    // }

    // __syncthreads();

    // // Perform the convolution operation on the tiles
    // if (x < W && y < H) {
        
    //     double sum = 0;
    //     for (int c = 0; c < C; c++) {
    //         for (int i = 0; i < FH; i++) {
    //             for (int j = 0; j < FW; j++) {
    //                 sum += d_F[k * C * FW * FH + c * FW * FH + FW * i + j] * input_tile[c][threadIdx.y + i][threadIdx.x + j];
    //             }
    //         }
    //     }
    //     d_O[(k * H * W) + x * H + y] = sum;
    // }

    extern __shared__ double shared_I0[];

    // Calculate the size of the shared memory block needed
    const int sharedMemWidth = blockDim.x + FW - 1;
    const int sharedMemHeight = blockDim.y + FH - 1;

    // Calculate the position in shared memory
    int k = blockIdx.z;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

     // Load input data into shared memory
    for (int c = 0; c < C; c++) {
        for (int i = threadIdx.x; i < sharedMemWidth; i += blockDim.x) {
            for (int j = threadIdx.y; j < sharedMemHeight; j += blockDim.y) {
                int global_x = blockIdx.x * blockDim.x + i;
                int global_y = blockIdx.y * blockDim.y + j;
                global_x = max(0, min(global_x, W + 2 * P - 1));
                global_y = max(0, min(global_y, H + 2 * P - 1));
                int shared_index = c * sharedMemHeight * sharedMemWidth + j * sharedMemWidth + i;
                int global_index = c * ((H + 2 * P) * (W + 2 * P)) + global_y * (W + 2 * P) + global_x;
                shared_I0[shared_index] = d_I0[global_index];
            }
        }
    }

    __syncthreads();

  // Perform convolution using shared memory
  if (x < W && y < H) {
    double sum = 0;
    for (int c = 0; c < C; c++) {
        for (int i = 0; i < FH; i++) {
            for (int j = 0; j < FW; j++) {
                int shared_x = threadIdx.x + i;
                int shared_y = threadIdx.y + j;
                int shared_index = c * sharedMemHeight * sharedMemWidth + shared_y * sharedMemWidth + shared_x;
                int filter_index = k * C * FH * FW + c * FH * FW + i * FW + j;
                sum += d_F[filter_index] * shared_I0[shared_index];
            }
        }
    }
    int output_index = (k * H * W) + x * H + y;
    d_O[output_index] = sum;
  }
}

int main() {
    int I_size = C * H * W;
    int F_size = K * C * FH * FW;
    int I0_size = C * (H + 2 * P) * (W + 2 * P);
    int O_size = K * W * H;

    double* h_I = new double[I_size];
    double* h_F = new double[F_size];
    double* h_I0 = new double[I0_size];
    double* h_O = new double[O_size];

    // Set all elements of I0 to zero
    memset(h_I0, 0, I0_size * sizeof(double));

     // Generate the elements of the I0, without padding
    for (int c = 0; c < C; c++) {
        for (int x = 0; x < H; x++) {
            for (int y = 0; y < W; y++) {
                int index_I = c * H * W + x * W  + y;
                h_I[index_I] = c * (x + y);
            }
        }
    }

    // Generate the elements of the filter F
    for (int k = 0; k < K; k++) {
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < FH; i++) {
                for (int j = 0; j < FW; j++) {
                    int index = k * (C * FH * FW) + c * (FH * FW) + i * FW + j;
                    h_F[index] = (c + k) * (i + j);
                }
            }
        }
    }

    // Generate the elements of the I0, with padding
    for (int c = 0; c < C; c++) {
        for (int x = 0; x < H; x++) {
            for (int y = 0; y < W; y++) {
                int index_I0 = c * ((H + 2 * P) * (W + 2 * P)) + (x + P) * (W + 2 * P) + (y + P);
                h_I0[index_I0] = c * (x + y);
            }
        }
    }

    //print_tensor("Tensor I", h_I, 1, C, H, W);
    //print_tensor("Tensor F", h_F, K, C, FH, FW);
    //print_tensor("Tensor I0", h_I0, 1, C, H + 2*P, W + 2*P);

    // allocate memory in GPU
    double* d_I0;
    double* d_F;
    double* d_O;

    cudaMalloc(&d_I0, I0_size * sizeof(double));
    cudaMalloc(&d_F, F_size * sizeof(double));
    cudaMalloc(&d_O, O_size * sizeof(double));

    // Copy I0 and F to GPU memory
    cudaMemcpy(d_I0, h_I0, I0_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, h_F, F_size * sizeof(double), cudaMemcpyHostToDevice);


    dim3 blockDim(32, 32);
    dim3 gridDim((W + blockDim.x - 1) / blockDim.x, (H + blockDim.y - 1) / blockDim.y, K);

    //Begin to run the kernel
    clock_t start_time = clock();
    
    convolution_kernel<<<gridDim, blockDim>>>(d_I0, d_F, d_O);
    cudaDeviceSynchronize();
    
    clock_t end_time = clock();
    double elapsed_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;

    //Get the output sum
    cudaMemcpy(h_O, d_O, O_size* sizeof(double), cudaMemcpyDeviceToHost);
    //print_tensor("Tensor O", h_O, 1, K, H, W );
    
    double checksum = check_sum(h_O, O_size);
    std::cout << "C1_checksum: " << checksum  << ", C1_execution_time: " << elapsed_time*1000 << " ms" << std::endl;

    cudaFree(d_O);

    //---------------------------------------C2 CODE------------------------------------------------
    cudaMalloc(&d_O, O_size * sizeof(double));

    blockDim = dim3(32, 32);
    gridDim = dim3((W + blockDim.x - 1) / blockDim.x, (H + blockDim.y - 1) / blockDim.y, K);

    const int sharedMemWidth = blockDim.x + FW - 1;
    const int sharedMemHeight = blockDim.y + FH - 1;
    int sharedMemSize = sharedMemWidth * sharedMemHeight * C * sizeof(double);

    //Begin to run the kernel
    start_time = clock();
    
    tiling_conv_kernel<<<gridDim, blockDim, sharedMemSize>>>(d_I0, d_F, d_O);
    cudaDeviceSynchronize();
    
    end_time = clock();
    elapsed_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;

    //Get the output sum
    cudaMemcpy(h_O, d_O, O_size* sizeof(double), cudaMemcpyDeviceToHost);
    //print_tensor("Tensor O_2", h_O, 1, K, H, W );
    
    checksum = check_sum(h_O, O_size);
    std::cout << "C2_checksum: " << checksum  << ", C2_execution_time: " << elapsed_time*1000 << " ms" << std::endl;

    cudaFree(d_O);
    cudaFree(d_I0);
    cudaFree(d_F);
    //---------------------------------------C3 CODE------------------------------------------------

    // allocate memory in GPU
    double* d_I;
    cudaMalloc(&d_I, I_size * sizeof(double));
    cudaMalloc(&d_F, F_size * sizeof(double));
    cudaMalloc(&d_O, O_size * sizeof(double));

    // Copy I0 and F to GPU memory
    cudaMemcpy(d_I, h_I, I_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, h_F, F_size * sizeof(double), cudaMemcpyHostToDevice);

    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    // Update the input tensor descriptor to use the padded tensor dimensions
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_DOUBLE,
                                        1, C, H, W));

    //Update the filter descriptor
    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                      CUDNN_DATA_DOUBLE,
                                      CUDNN_TENSOR_NCHW,
                                      K, C, FH, FW));
    
    //Update the convolution descriptor
    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                               P, P, 1, 1,
                                               1, 1,
                                               CUDNN_CONVOLUTION,
                                               CUDNN_DATA_DOUBLE));
    
    //Update the output descriptor
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_DOUBLE,
                                          1, K, H, W));
    
    //Find the most suitable forward Algorithm
    const int requested_algo_count = 1;
    int returned_algo_count;
    cudnnConvolutionFwdAlgoPerf_t perf_results;
    checkCUDNN(cudnnFindConvolutionForwardAlgorithm(cudnn,
                                                    input_descriptor,
                                                    kernel_descriptor,
                                                    convolution_descriptor,
                                                    output_descriptor,
                                                    requested_algo_count,
                                                    &returned_algo_count,
                                                    &perf_results));

    cudnnConvolutionFwdAlgo_t convolution_algorithm = perf_results.algo;
    //std::cout << "Best algorithm: " << convolution_algorithm << std::endl;

    //Find the workspace
    size_t workspace_bytes{0};
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                    input_descriptor,
                                                    kernel_descriptor,
                                                    convolution_descriptor,
                                                    output_descriptor,
                                                    convolution_algorithm,
                                                    &workspace_bytes));

    // Allocate the workspace on the GPU
    void* d_workspace;
    cudaMalloc(&d_workspace, workspace_bytes);
    
    //Perform convolution
    double alpha = 1.0;
    double beta = 0.0;
    
    start_time = clock();
    checkCUDNN(cudnnConvolutionForward(cudnn,
                                    &alpha,
                                    input_descriptor, d_I,
                                    kernel_descriptor, d_F,
                                    convolution_descriptor, convolution_algorithm,
                                    d_workspace, workspace_bytes,
                                    &beta,
                                    output_descriptor, d_O));
    cudaDeviceSynchronize();
    end_time = clock();
    elapsed_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;


    //Get the output sum
    cudaMemcpy(h_O, d_O, O_size* sizeof(double), cudaMemcpyDeviceToHost);
    //print_tensor("Tensor O", h_O, 1, K, H, W );
    
    
    checksum = check_sum(h_O, O_size);
    std::cout << "C3_checksum: " << checksum  << ", C3_execution_time: " << elapsed_time*1000 << " ms" << std::endl;


    //Free memory
    delete[] h_I;
    delete[] h_I0;
    delete[] h_F;
    delete[] h_O;


    cudaFree(d_I);
    cudaFree(d_F);
    cudaFree(d_O);
    cudaFree(d_workspace);

    // Destroy the input descriptor
    checkCUDNN(cudnnDestroyTensorDescriptor(input_descriptor));

    // Destroy the filter descriptor (kernel descriptor)
    checkCUDNN(cudnnDestroyFilterDescriptor(kernel_descriptor));

    // Destroy the output descriptor
    checkCUDNN(cudnnDestroyTensorDescriptor(output_descriptor));

    // Destroy the convolution descriptor
    checkCUDNN(cudnnDestroyConvolutionDescriptor(convolution_descriptor));

    // Destroy the cuDNN handle
    checkCUDNN(cudnnDestroy(cudnn));

    return 0;
}
