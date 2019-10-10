#define GLM_FORCE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include "svd3.h"
#include "GPU_kernel.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <glm/glm.hpp>
#include "utilities.h"
//#include "kernel.h"
#include <thrust/reduce.h>
#include <glm/vec3.hpp>



#define blockSize 128



glm::vec3* dev_first;
glm::vec3* dev_first_buf;
glm::vec3* dev_second;
glm::vec3* dev_corr;
glm::mat3* dev_rot;
glm::vec3* dev_trans;



void scanmatch::GPU::initSimulation(int N_first, int N_second, glm::vec3 *first, glm::vec3 *second) {

	cudaMalloc((void**)&dev_first, N_first * sizeof(glm::vec3));
	cudaMemcpy(dev_first, first, sizeof(glm::vec3) * N_first, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_first, N_first * sizeof(glm::vec3));
	cudaMemcpy(dev_first_buf, first, sizeof(glm::vec3) * N_first, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_second, N_second * sizeof(glm::vec3));
	cudaMemcpy(dev_second, second, sizeof(glm::vec3) * N_second, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_corr, N_first * sizeof(glm::vec3));

	cudaMalloc((void**)&dev_rot, sizeof(glm::mat3));

	cudaMalloc((void**)&dev_trans, sizeof(glm::vec3));

}




/*
// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(cublasHandle_t &handle, const float *A, const float *B, float *C, const int m, const int k, const int n) {
	int lda = m, ldb = k, ldc = m;
	 float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;

	// Do the actual multiplication
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
*/
/*
__global__ void matrix_subtraction(float* A, float* B, float* C, int m, int n) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= m*n) {
		return;
	}
	C[index] = A[index] - B[index];
}

__global__ void addTranslation(float* A, float* trans, int num) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num) {
		return;
	}
	A[index * 3 + 0] += trans[0];
	A[index * 3 + 1] += trans[1];
	A[index * 3 + 2] += trans[2];
}

__global__ void setValueOnDevice(float* device_var, int val) {
	*device_var = val;
}


__global__ void find_svd(glm::mat3 &w, glm::mat3 &u, glm::mat3 &s, glm::mat3 &v) {

	svd(w[0][0], w[0][1], w[0][2], w[1][0], w[1][1], w[1][2], w[2][0], w[2][1], w[2][2],
		u[0][0], u[0][1], u[0][2], u[1][0], u[1][1], u[1][2], u[2][0], u[2][1], u[2][2],
		s[0][0], s[0][1], s[0][2], s[1][0], s[1][1], s[1][2], s[2][0], s[2][1], s[2][2],
		v[0][0], v[0][1], v[0][2], v[1][0], v[1][1], v[1][2], v[2][0], v[2][1], v[2][2]);
}
*/
void scanmatch::GPU::run(int N_first, int N_second) {

	dim3 numBlocks_first((N_first + blockSize - 1) / blockSize);
	dim3 numBlocks_second((N_second + blockSize - 1) / blockSize);
	dim3 numBlocks_rot((3 * 3 + blockSize - 1) / blockSize);
	dim3 numBlocks3_tran((3 * 1 + blockSize - 1) / blockSize);
	
	findmatch << <numBlocks_first, blockSize >> > (N_first, N_second, dev_first, dev_second, dev_corr);

	glm::vec3 *dev_mean_first;
	glm::vec3 *dev_mean_corr;

	cudaMalloc((void**)&dev_mean_first, sizeof(glm::vec3));
	cudaMalloc((void**)&dev_mean_corr, sizeof(glm::vec3));

	find_mean_vec(N_first, dev_first, dev_mean_first);
	find_mean_vec(N_first, dev_corr, dev_mean_corr);
	
	glm::vec3 *dev_centered_first;
	glm::vec3 *dev_centered_corr;

	cudaMalloc((void**)&dev_centered_first, N_first* sizeof(glm::vec3));
	cudaMalloc((void**)&dev_centered_corr, N_first* sizeof(glm::vec3));

	cudaMemcpy(dev_centered_first, dev_first, N_first * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_centered_corr, dev_corr, N_first * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);

	Subtract_element << <numBlocks_first, blockSize >> > (N_first, dev_centered_first, dev_mean_first);
	Subtract_element << <numBlocks_first, blockSize >> > (N_first, dev_centered_corr, dev_mean_corr);

	glm::mat3 *dev_B_svds;

	cudaMalloc((void**)&dev_B_svds, N_first * sizeof(glm::mat3));

	multiply_transpose << <numBlocks_first, blockSize >> > (N_first, dev_centered_first, dev_centered_corr, dev_B_svds);

	//glm::mat3 *dev_W;

	//cudaMalloc((void**)&dev_W, sizeof(glm::mat3));
	
	glm::mat3 W = thrust::reduce(thrust::device, dev_B_svds, dev_B_svds + N_first, glm::mat3(0));

	//glm::mat3 *dev_W;
	//glm::mat3 U;
	//glm::mat3 S;
	//glm::mat3 V;

	//cudaMalloc((void**)&dev_W, sizeof(glm::mat3));

	//cudaMemcpy(dev_W, &W, sizeof(glm::mat3), cudaMemcpyHostToDevice);

	float U[3][3] = { 0 };
	float S[3][3] = { 0 };
	float V[3][3] = { 0 };
	
	svd(W[0][0], W[0][1], W[0][2], W[1][0], W[1][1], W[1][2], W[2][0], W[2][1], W[2][2],
		U[0][0], U[0][1], U[0][2], U[1][0], U[1][1], U[1][2], U[2][0], U[2][1], U[2][2],
		S[0][0], S[0][1], S[0][2], S[1][0], S[1][1], S[1][2], S[2][0], S[2][1], S[2][2],
		V[0][0], V[0][1], V[0][2], V[1][0], V[1][1], V[1][2], V[2][0], V[2][1], V[2][2]);


	glm::mat3 host_U(glm::vec3(U[0][0], U[1][0], U[2][0]), glm::vec3(U[0][1], U[1][1], U[2][1]), glm::vec3(U[0][2], U[1][2], U[2][2]));
	glm::mat3 host_Vt(glm::vec3(V[0][0], V[0][1], V[0][2]), glm::vec3(V[1][0], V[1][1], V[1][2]), glm::vec3(V[2][0], V[2][1], V[2][2]));

	glm::vec3 *host_mean_first = new glm::vec3[1];
	glm::vec3 *host_mean_corr = new glm::vec3[1];

	cudaMemcpy(host_mean_first, dev_mean_first, sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_mean_corr, dev_mean_corr, sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	glm::mat3 host_rot = host_U * host_Vt;
	glm::vec3 host_trans = host_mean_corr[0] - host_rot * host_mean_first[0];

	cudaMemcpy(dev_rot, &host_rot, sizeof(glm::mat3) , cudaMemcpyHostToDevice);
	cudaMemcpy(dev_trans, &host_trans, sizeof(glm::mat3), cudaMemcpyHostToDevice);


	update << <numBlocks_first, blockSize >> > (N_first, dev_first, *dev_rot, *dev_trans, dev_first_buf);

	cudaMemcpy(dev_first, dev_first_buf, N_first * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
}

