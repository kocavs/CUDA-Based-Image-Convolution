# CUDA-Based Image Convolution

## Project Overview

This project implements image convolution using CUDA, demonstrating three distinct methods to leverage GPU acceleration for processing images with convolution filters. It showcases basic CUDA operations, tiled convolution with shared memory, and utilizes NVIDIA's cuDNN library for optimized convolution operations. The focus is on comparing performance in terms of execution time and accuracy (checksum verification) across these methods.

### Key Features:

- Utilizes CUDA for GPU-accelerated image processing.
- Implements tiled convolution using shared memory for efficient GPU resource utilization.
- Employs NVIDIA's cuDNN library for high-performance convolution operations.
- Compares performance metrics across different convolution methods.

## Prerequisites

- NVIDIA CUDA Toolkit (recommended version 10.2 or later)
- NVIDIA cuDNN library (compatible with the CUDA version)
- C++ compiler with support for the CUDA Toolkit
- An NVIDIA GPU with CUDA Compute Capability 5.0 or higher

## Installation

1. Ensure the CUDA Toolkit and cuDNN library are installed on your system.
2. Clone the repository to your local machine
3. Navigate to the cloned directory.

## Usage

To compile the project, run the following command in the terminal:
```
nvcc -o image_convolution c1.cu -lcudnn
```
Execute the compiled program with:
```
./image_convolution
```

## Results

The convolution operations yield the following performance metrics:

- Basic CUDA operations: Execution time of 22.225 ms
- Tiled convolution and shared memory utilization: Execution time of 14.427 ms
- Utilization of NVIDIA's cuDNN library: Execution time of 13.022 ms

All methods achieve the same checksum value, ensuring consistency and accuracy across the different convolution implementations.

## Acknowledgments

This project was developed as part of a lab exercise in the CUDA programming course. Special thanks to NVIDIA for providing the CUDA Toolkit and cuDNN library, facilitating high-performance GPU computations.
