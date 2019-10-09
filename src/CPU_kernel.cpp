#define GLM_FORCE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <cublas_v2.h>
#include <fstream>
#include <glm/glm.hpp>
#include "svd3.h"
#include "kernel.h"
#include "device_launch_parameters.h"
#include "CPU_kernel.h"


#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)


void checkCUDAError(const char *msg, int line = -1) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		if (line >= 0) {
			fprintf(stderr, "Line %d: ", line);
		}
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

#define blockSize 128


glm::vec3* dev_first;
glm::vec3* dev_second;

glm::vec3 findmatch(int N_first, int N_second, glm::vec3* first, glm::vec3* second) {
	
	glm::vec3 *corr;
	
	glm::vec3 *corr = new glm::vec3[N_first];
	

	glm::vec3 desired_point;
	float min_distance = LONG_MAX;

	for (int i = 0; i < N_first; i++) {
		for (int j = 0; j < N_second; j++) {
			float distance = glm::distance(first[i], second[j]);
			if (distance < min_distance) {
				desired_point = second[j];
				min_distance = distance;
			}

		}
		corr[i] = desired_point;
	}
	

	return *corr;
}

void find_mean_and_sub(int n, glm::vec3 *idata, glm::vec3 *host_mean, glm::vec3 *host_centered) {

	*host_centered = *idata;

	for (int i = 0; i < n; i++) {
		
		*host_mean += idata[i];
	}
	
	*host_mean /= n;



	for (int i = 0; i < n; i++) {
		host_centered[i] -= host_mean;
	}
	
}

glm::mat3 multiply_transpose(int n, glm::vec3* first, glm::vec3* corr) {
	
	glm::mat3 W;

	float a, b;

	float sum = 0.0f;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			sum = 0;
			for (int k = 0; k < n; k++) {
				a = (i == 0 ? corr[k].x : i == 1 ? corr[k].y : corr[k].z);
				b = (j == 0 ? first[k].x : j == 1 ? first[k].y : first[k].z);
				sum += a * b;
			}
			W[i][j] = sum;
		}
	}


	return W;
}

glm::vec3 update(int N_first, glm::vec3 *host_first, glm::mat3 dev_rot, glm::vec3 dev_trans) {
	
	glm::vec3 *host_first_buf;

	glm::vec3 *host_first_buf = new glm::vec3[N_first];

	*host_first_buf = *host_first;

	for (int i = 0; i < N_first; i++) {
		host_first_buf[i] = (dev_rot * host_first[i]) + dev_trans;
	}
	
	
	return *host_first_buf;


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

void scanmatch::CPU::initSimulation(int N_first, int N_second, glm::vec3* first, glm::vec3* second) {
	cudaMalloc((void**)&dev_first, N_first * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_first failed!");
	cudaMemcpy(dev_first, first, sizeof(glm::vec3) * N_first, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_second, N_second * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_second failed!");
	cudaMemcpy(dev_second, second, sizeof(glm::vec3) * N_second, cudaMemcpyHostToDevice);

}
void scanmatch::CPU::run(int N_first, int N_second, glm::vec3* first_points, glm::vec3* second_points) {


	glm::vec3 host_corr = findmatch(N_first, N_second, first_points, second_points);
	
	glm::vec3 host_mean_first(0.0f, 0.0f, 0.0f);
	glm::vec3 host_mean_corr(0.0f, 0.0f, 0.0f);
	//glm::vec3 *mean = new glm::vec3[1];
	glm::vec3 *host_centered_first;
	glm::vec3 *host_centered_first = new glm::vec3[N_first];
	//*host_centered_first = *first_points;

	glm::vec3 *host_centered_corr;
	glm::vec3 *host_centered_corr = new glm::vec3[N_first];
	//*host_centered_corr = host_corr;
	
	find_mean_and_sub(N_first, first_points, &host_mean_first, host_centered_first);
	find_mean_and_sub(N_first, &host_corr, &host_mean_corr, host_centered_corr);

	glm::mat3 W = multiply_transpose(N_first, host_centered_first, host_centered_corr);

	glm::mat3 U;
	glm::mat3 S;
	glm::mat3 V;

	svd(W[0][0], W[0][1], W[0][2], W[1][0], W[1][1], W[1][2], W[2][0], W[2][1], W[2][2],
		U[0][0], U[0][1], U[0][2], U[1][0], U[1][1], U[1][2], U[2][0], U[2][1], U[2][2],
		S[0][0], S[0][1], S[0][2], S[1][0], S[1][1], S[1][2], S[2][0], S[2][1], S[2][2],
		V[0][0], V[0][1], V[0][2], V[1][0], V[1][1], V[1][2], V[2][0], V[2][1], V[2][2]);

	glm::mat3 host_U(glm::vec3(U[0][0], U[1][0], U[2][0]), glm::vec3(U[0][1], U[1][1], U[2][1]), glm::vec3(U[0][2], U[1][2], U[2][2]));
	glm::mat3 host_Vt(glm::vec3(V[0][0], V[0][1], V[0][2]), glm::vec3(V[1][0], V[1][1], V[1][2]), glm::vec3(V[2][0], V[2][1], V[2][2]));

	glm::mat3 host_rot = host_U * host_Vt;
	glm::vec3 host_trans = host_mean_corr - host_rot * host_mean_first;

	glm::vec3 new_buf =  update(N_first, first_points, host_rot, host_trans);

	cudaMemcpy(first_points, &new_buf, N_first * sizeof(glm::vec3), cudaMemcpyHostToHost);
}


void scanmatch::CPU::host_to_dev(int N_first, int N_second, glm::vec3* first_points, glm::vec3* second_points) {

	cudaMemcpy(dev_first, first_points, N_first * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_second, second_points, N_first * sizeof(glm::vec3), cudaMemcpyHostToDevice);
}