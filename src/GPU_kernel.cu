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
#include "GPU_kernel.h"


#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

#define blockSize 128



glm::vec3* dev_first;
glm::vec3* dev_second;
glm::vec3* dev_corr;
glm::mat3* dev_rot;
glm::vec3* dev_trans;


void scanmatch::GPU::initSimulation(int N_first, int N_second,float *first,float *second) {

	cudaMalloc((void**)&dev_first, N_first * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_first failed!");
	cudaMemcpy(dev_first, first, sizeof(glm::vec3) * N_first, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_second, N_second * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_second failed!");
	cudaMemcpy(dev_second, second, sizeof(glm::vec3) * N_second, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_corr, N_first * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_corr failed!");

	cudaMalloc((void**)&dev_rot, sizeof(glm::mat3));
	checkCUDAErrorWithLine("cudaMalloc dev_rot failed!");

	cudaMalloc((void**)&dev_trans, sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_trans failed!");

}



__global__ void findmatch(int N_first, int N_second, glm::vec3* dev_first, glm::vec3* dev_second, glm::vec3* dev_corr) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= N_first) {
		return;
	}
	
	glm::vec3 desired_point;
	float min_distance = LONG_MAX;
	for (int ind = 0; ind < N_second; ind++) {
		float distance = glm::distance(dev_first[index], dev_second[ind]);
		if (distance < min_distance) {
			desired_point = dev_second[ind];
			min_distance = distance;
		}

	}
	
	dev_corr[index] = desired_point;
}

__global__ void up_sweep(int N, glm::vec3 *Dev_odata, int d) {

	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	index = index * (1 << (d + 1));

	if (index > N - 1) {
		return;
	}

	if (((index + (1 << (d)) - 1) < N) && ((index + (1 << (d + 1)) - 1) < N)) {

		Dev_odata[index + (1 << (d + 1)) - 1].x += Dev_odata[index + (1 << (d)) - 1].x;
		Dev_odata[index + (1 << (d + 1)) - 1].y += Dev_odata[index + (1 << (d)) - 1].y;
		Dev_odata[index + (1 << (d + 1)) - 1].z += Dev_odata[index + (1 << (d)) - 1].z;
	}

}

inline int ilog2(int x) {
	int lg = 0;
	while (x >>= 1) {
		++lg;
	}
	return lg;
}

inline int ilog2ceil(int x) {
	return x == 1 ? 0 : ilog2(x - 1) + 1;
}

void find_mean_vec(int n, glm::vec3 *dev_idata, glm::vec3 *dev_mean) {

	//printArray(n, idata);
	//int new_n = n;
	n = 1 << ilog2ceil(n); // make n something that is power of 2
	
	glm::vec3 *dev_odata;
	cudaMalloc((void**)&dev_odata, n * sizeof(glm::vec3));

	cudaMemcpy(dev_odata, dev_idata, n * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);

	for (int d = 0; d <= ((ilog2ceil(n)) - 1); d++) {
		int count_thread = 1 << ((ilog2ceil(n) - d - 1));   // i need ceil(n/d) threads total
		dim3 fullBlocksPerGrid(((count_thread)+blockSize - 1) / blockSize);
		up_sweep << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, d);
	}
	dev_odata[n - 1] /= n;
	dev_mean = &dev_odata[n - 1];
}

__global__ void Subtract_element(int n,glm::vec3* dev_idata, glm::vec3* mean) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index > N - 1) {
		return;
	}

	dev_idata[index] -= mean;
}

__global__ void multiply_transpose(int n, glm::vec3* dev_first, glm::vec3* dev_second, glm::mat3 *out) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index > N - 1) {
		return;
	}

	out[index] = glm::outerProduct(dev_first[index], dev_second[index]);
}

__global__ void update(int N_first, glm::vec3 *, glm::mat3 R, glm::vec3 trans) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N_first)
		return;

	dev_pos[index] = R * dev_pos[index] + trans;
}

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

	multiply_transpose << <numBlocks_first, blockSize >> > (N_first, dev_mean_first, dev_mean_corr, dev_B_svds);

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

	glm::mat3 R = host_U * host_Vt;
	glm::vec3 t = host_mean_corr[0] - R * host_mean_first[0];

}

















































void getArraySum(int n, float* input, float* sum) {
	float* padded_idata;
	int padded_size = 1 << (ilog2ceil(n));

	cudaMalloc((void**)&padded_idata, padded_size * sizeof(float));
	checkCUDAErrorWithLine("cudaMalloc padded_idata failed!");

	cudaMemset(padded_idata, 0, padded_size * sizeof(float));
	cudaMemcpy(padded_idata, input, sizeof(float) * n, cudaMemcpyDeviceToDevice);

	int iterations = ilog2(padded_size);

	int number_of_threads = padded_size;
	for (int d = 0; d < iterations; d++) {
		number_of_threads /= 2;
		dim3 fullBlocksPerGridUpSweep((number_of_threads + blockSize - 1) / blockSize);
		upSweepOptimized << <fullBlocksPerGridUpSweep, blockSize >> >(padded_size, d, padded_idata);
	}

	cudaMemcpy(sum, padded_idata + padded_size - 1, sizeof(float), cudaMemcpyDeviceToDevice);

	cudaFree(padded_idata);
}




























void printMatrix(float* A, int m, int n) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			std::cout << A[i*n + j] << " ";
		}
		std::cout << std::endl;
	}
}


































namespace NaiveGPU {
	float* dev_x;
	float* dev_y;

	float* dev_x_corr;
	float* dev_R;
	float* dev_translation;


	void initScan(int numX) {


		cudaMalloc((void**)&dev_x_corr, numX * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_x_corr failed!");

		cudaMalloc((void**)&dev_R, 3 * 3 * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_R failed!");

		cudaMalloc((void**)&dev_translation, 3 * 1 * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_translation failed!");
	}

	void match(float* x, float* y, int numX, int numY) {

		int eachX = numX / 3;
		int eachY = numY / 3;

		dim3 numBlocks((eachX + blockSize - 1) / blockSize);
		dim3 numBlocks1((numX+blockSize - 1) / blockSize);
		dim3 numBlocks2((3 * 3 + blockSize - 1) / blockSize);
		dim3 numBlocks3((3 * 1 + blockSize - 1) / blockSize);

		//Copy data to GPU
		cudaMalloc((void**)&dev_x, numX * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_x failed!");
		cudaMemcpy(dev_x, x, sizeof(float) * numX, cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_y, numY * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_y failed!");
		cudaMemcpy(dev_y, y, sizeof(float) * numY, cudaMemcpyHostToDevice);

		//Find Correspondence
		findCorrespondence << <numBlocks, blockSize >> >(dev_x, numX, dev_y, numY, dev_x_corr);

		//Transpose x_corr and x
		float* dev_x_tr;
		cudaMalloc((void**)&dev_x_tr, numX * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_x failed!");
		transpose << <numBlocks1, blockSize >> >(dev_x, dev_x_tr, eachX, 3);

		float* dev_x_corr_tr;
		cudaMalloc((void**)&dev_x_corr_tr, numX * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_x failed!");
		transpose << <numBlocks1, blockSize >> >(dev_x_corr, dev_x_corr_tr, eachX, 3);

		//Mean-center x
		float* meanX;
		cudaMalloc((void**)&meanX, sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc sum failed!");
		getArraySum(eachX, dev_x_tr, meanX);

		float* meanY;
		cudaMalloc((void**)&meanY, sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc sum failed!");
		getArraySum(eachX, dev_x_tr + eachX, meanY);

		float* meanZ;
		cudaMalloc((void**)&meanZ, sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc sum failed!");
		getArraySum(eachX, dev_x_tr + (eachX * 2), meanZ);

		cudaFree(dev_x_tr);

		//Mean-center x_corr
		float* meanXC;
		cudaMalloc((void**)&meanXC, sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc sum failed!");
		getArraySum(eachX, dev_x_corr_tr, meanXC);

		float* meanYC;
		cudaMalloc((void**)&meanYC, sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc sum failed!");
		getArraySum(eachX, dev_x_corr_tr + eachX, meanYC);

		float* meanZC;
		cudaMalloc((void**)&meanZC, sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc sum failed!");
		getArraySum(eachX, dev_x_corr_tr + (eachX * 2), meanZC);
		
		cudaFree(dev_x_corr_tr);

		meanCenter <<<numBlocks, blockSize >>>(dev_x, eachX, *meanX, *meanY, *meanZ);
		meanCenter <<<numBlocks, blockSize >>>(dev_x_corr, eachX, *meanXC, *meanYC, *meanZC);

		//Multiply x_corr_tr and x to get input to SVD
		cudaMalloc((void**)&dev_x_corr_tr, numX * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_x failed!");
		transpose << <numBlocks1, blockSize >> > (dev_x_corr, dev_x_corr_tr, eachX, 3);

		float* dev_to_svd;
		cudaMalloc((void**)&dev_to_svd, 3 * 3 * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_to_svd failed!");

		// Create a handle for CUBLAS
		cublasHandle_t handle;
		cublasCreate(&handle);
		gpu_blas_mmul(handle, dev_x_corr_tr, dev_x, dev_to_svd, 3, eachX, 3);

		//Find SVD - U, V, S
		float* dev_svd_u;
		cudaMalloc((void**)&dev_svd_u, 3 * 3 * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_to_svd failed!");

		float* dev_svd_s;
		cudaMalloc((void**)&dev_svd_s, 3 * 3 * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_to_svd failed!");

		float* dev_svd_v;
		cudaMalloc((void**)&dev_svd_v, 3 * 3 * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_to_svd failed!");

		get_svd << <1, 1 >> > (dev_to_svd, dev_svd_u, dev_svd_s, dev_svd_v);

		cudaFree(dev_svd_s);
		//Compute U x V_tr to get R
		float* dev_svd_v_tr;
		cudaMalloc((void**)&dev_svd_v_tr, 3 * 3 * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_to_svd failed!");
		transpose << <numBlocks2, blockSize >> > (dev_svd_v, dev_svd_v_tr, 3, 3);

		cudaFree(dev_svd_v);

		gpu_blas_mmul(handle, dev_svd_u, dev_svd_v_tr, dev_R, 3, 3, 3);

		//Compute translation = x_corr_mean - R.x_mean
		float* dev_x_mean;
		cudaMalloc((void**)&dev_x_mean, 3 * sizeof(float));
		dev_x_mean[0] = *meanX;
		dev_x_mean[1] = *meanY;
		dev_x_mean[2] = *meanZ;

		float* dev_y_mean;
		cudaMalloc((void**)&dev_y_mean, 3 * sizeof(float));
		dev_y_mean[0] = *meanXC;
		dev_y_mean[1] = *meanYC;
		dev_y_mean[2] = *meanZC;

		float* inter;
		cudaMalloc((void**)&inter, 3 * 1 * sizeof(float));
		gpu_blas_mmul(handle, dev_R, dev_x_mean, inter, 3, 3, 1);

		matrix_subtraction << <numBlocks3, blockSize >> > (dev_y_mean, inter, dev_translation, 1, 3);

		//Apply rotation on x
		float* dev_newX;
		cudaMalloc((void**)&dev_newX, numX * sizeof(float));
		gpu_blas_mmul(handle, dev_x, dev_R, dev_newX, eachX, 3, 3);

		//Apply translation on x
		addTranslation << <numBlocks, blockSize >> > (dev_newX, dev_translation, eachX);

		cudaMemcpy(x, dev_newX, sizeof(float) * numX, cudaMemcpyDeviceToHost);
	}
}

