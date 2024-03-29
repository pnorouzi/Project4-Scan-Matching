#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
//#include "utilityCore.hpp"
#include "kernel.h"
#include "device_launch_parameters.h"
#include <glm/gtc/type_ptr.hpp>
#include "svd3.h"
#include <time.h>

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

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

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
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


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128
#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 0.1f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

clock_t timer;
double time_per_iter = 0;


glm::vec3 *dev_pos;
glm::vec3 *dev_color;


glm::vec3* dev_first;

glm::vec3* dev_second;
glm::vec3* dev_corr;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
	thrust::default_random_engine rng(hash((int)(index * time)));
	thrust::uniform_real_distribution<float> unitDistrib(-1, 1);
	return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));

}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void AddColor(int N, glm::vec3* dev_color, glm::vec3 val) {

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		dev_color[index] = val;
	}
}


__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	float c_scale = -1.0f / s_scale;

	if (index < N) {
		vbo[4 * index + 0] = pos[index].x * c_scale;
		vbo[4 * index + 1] = pos[index].y * c_scale;
		vbo[4 * index + 2] = pos[index].z * c_scale;
		vbo[4 * index + 3] = 1.0f;
	}
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index < N) {
		vbo[4 * index + 0] = vel[index].x + 0.3f;
		vbo[4 * index + 1] = vel[index].y + 0.3f;
		vbo[4 * index + 2] = vel[index].z + 0.3f;
		vbo[4 * index + 3] = 1.0f;
	}
}

void scanmatch::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

	kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos, vbodptr_positions, scene_scale);
	kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_color, vbodptr_velocities, scene_scale);

	checkCUDAErrorWithLine("copyBoidsToVBO failed!");

	cudaDeviceSynchronize();
}



/**
* Initialize memory, update some globals
*/
void scanmatch::initSimulation(int N_first, int N_second, glm::vec3* first_points, glm::vec3* second_points) {
	int N = N_first + N_second;
	numObjects = N;

	cudaMalloc((void**)&dev_pos, numObjects * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

	cudaMalloc((void**)&dev_color, numObjects * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_color failed!");

	cudaMalloc((void**)&dev_first, N_first * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc failed!");
	cudaMemcpy(dev_first, first_points, N_first * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("cudaMemcpy failed!");


	cudaMalloc((void**)&dev_second, N_second * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc failed!");
	cudaMemcpy(dev_second, second_points, sizeof(glm::vec3) * N_second, cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("cudaMemcpy failed!");

	cudaMalloc((void**)&dev_corr, N_first * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc failed!");


	cudaMemcpy(dev_pos, first_points, N_first * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("cudaMemcpy failed!");
	cudaMemcpy(dev_pos + N_first, second_points, N_second * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("cudaMemcpy failed!");

	

	dim3 fullBlocksPerGrid1((N_first + blockSize - 1) / blockSize);
	AddColor << <fullBlocksPerGrid1, blockSize >> > (N_first, dev_color, glm::vec3(0, 0.8, 0.4));

	dim3 fullBlocksPerGrid2((N_second + blockSize - 1) / blockSize);
	AddColor << <fullBlocksPerGrid2, blockSize >> > (N_second, dev_color+N_first, glm::vec3(1, 0.08, 0.6));

	cudaDeviceSynchronize();
}



__global__ void findmatch(int N_first, int N_second, glm::vec3* dev_first, glm::vec3* dev_second, glm::vec3* dev_corr) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;


	if (index >= N_first) {
		return;
	}

	glm::vec3 desired_point;

	float min_distance = glm::distance(dev_first[index], dev_second[0]);
	desired_point = dev_second[0];

	for (int ind = 1; ind < N_second; ind++) {
		
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

void find_mean_vec(int n, glm::vec3 *dev_idata, glm::vec3 *dev_sum) {

	//printArray(n, idata);
	//int new_n = n;
	n = 1 << ilog2ceil(n); // make n something that is power of 2
	//printf("here \n");
	glm::vec3 *dev_odata;
	cudaMalloc((void**)&dev_odata, n * sizeof(glm::vec3));
	//printf("here \n");
	cudaMemcpy(dev_odata, dev_idata, n * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
	//printf("here \n");
	for (int d = 0; d <= ((ilog2ceil(n)) - 1); d++) {
		//printf("here \n");
		int count_thread = 1 << ((ilog2ceil(n) - d - 1));   // i need ceil(n/d) threads total
		dim3 fullBlocksPerGrid(((count_thread)+blockSize - 1) / blockSize);
		//printf("here \n");
		up_sweep << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, d);
		//printf("here \n");
	}
	//printf("here \n");
	cudaMemcpy(dev_sum, dev_odata+n-1, sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
	//printf("here \n");
	//printf("%f,%f,%f\n", dev_odata[n - 1].x, dev_odata[n - 1].y, dev_odata[n - 1].z);
	//printf("here \n");
}

__global__ void multiply_transpose(int n, glm::vec3* dev_first, glm::vec3* dev_second, glm::vec3 dev_mean_first, glm::vec3 dev_mean_corr, glm::mat3 *out) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index > n - 1) {
		return;
	}

	out[index] = glm::outerProduct(dev_first[index]- dev_mean_first, dev_second[index]- dev_mean_corr);
}

__global__ void update(int N_first, glm::vec3 *dev_first, glm::mat3 dev_rot, glm::vec3 dev_trans) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N_first)
		return;

	dev_first[index] = (dev_rot * dev_first[index]) + dev_trans;
}

void scanmatch::run_GPU(int N_first, int N_second) {
	
	timer = clock();

	dim3 numBlocks_first((N_first + blockSize - 1) / blockSize);

	findmatch << <numBlocks_first, blockSize >> > (N_first, N_second, dev_first, dev_second, dev_corr);
	//printf("here \n");
	thrust::device_ptr<glm::vec3> thrust_dev_first(dev_first);
	thrust::device_ptr<glm::vec3> thrust_dev_correspond(dev_corr);
	//printf("here \n");
	glm::vec3 *dev_mean_first;
	glm::vec3 *dev_mean_corr;

	cudaMalloc((void**)&dev_mean_first, sizeof(glm::vec3));
	cudaMalloc((void**)&dev_mean_corr, sizeof(glm::vec3));


	glm::vec3 mean_first = glm::vec3(thrust::reduce(thrust_dev_first, thrust_dev_first + N_first, glm::vec3(0.0f, 0.0f, 0.0f)));
	glm::vec3 mean_corr = glm::vec3(thrust::reduce(thrust_dev_correspond, thrust_dev_correspond + N_first, glm::vec3(0.0f, 0.0f, 0.0f)));


	mean_first /= float(N_first);
	mean_corr /= float(N_first);


	glm::mat3 *dev_B_svds;

	cudaMalloc((void**)&dev_B_svds, N_first * sizeof(glm::mat3));

	multiply_transpose << <numBlocks_first, blockSize >> > (N_first, dev_first, dev_corr, mean_first, mean_corr, dev_B_svds);

	
	glm::mat3 W = thrust::reduce(thrust::device, dev_B_svds, dev_B_svds + N_first, glm::mat3(0));
	
	float U[3][3] = { 0 };
	float S[3][3] = { 0 };
	float V[3][3] = { 0 };
	//printf("%f \n", V[3][3]);
	
	svd(W[0][0], W[0][1], W[0][2], W[1][0], W[1][1], W[1][2], W[2][0], W[2][1], W[2][2],
		U[0][0], U[0][1], U[0][2], U[1][0], U[1][1], U[1][2], U[2][0], U[2][1], U[2][2],
		S[0][0], S[0][1], S[0][2], S[1][0], S[1][1], S[1][2], S[2][0], S[2][1], S[2][2],
		V[0][0], V[0][1], V[0][2], V[1][0], V[1][1], V[1][2], V[2][0], V[2][1], V[2][2]);


	glm::mat3 host_U = glm::mat3(glm::vec3(U[0][0], U[1][0], U[2][0]), glm::vec3(U[0][1], U[1][1], U[2][1]), glm::vec3(U[0][2], U[1][2], U[2][2]));

	glm::mat3 host_rot = host_U * glm::mat3(glm::vec3(V[0][0], V[0][1], V[0][2]), glm::vec3(V[1][0], V[1][1], V[1][2]), glm::vec3(V[2][0], V[2][1], V[2][2]));
	

	
	glm::vec3 host_trans = mean_corr - (host_rot* mean_first);

	
	update << <numBlocks_first, blockSize >> > (N_first, dev_first, host_rot, host_trans);
	//update << <numBlocks_first, blockSize >> > (N_first, dev_first, R, T, dev_pos);


	timer = clock() - timer;
	time_per_iter = ((double)timer) / CLOCKS_PER_SEC;
	
	printf("(Time per Iter : %f \n", time_per_iter);


	cudaMemcpy(dev_pos, dev_first, N_first * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
	checkCUDAErrorWithLine("cudaMemcpy failed!");
	cudaDeviceSynchronize();
}






void findmatch_cpu(int N_first, int N_second, glm::vec3* first, glm::vec3* second, glm::vec3* corr) {

	glm::vec3 desired_point;
	for (int i = 0; i < N_first; i++) {
		float min_distance = glm::distance(first[i], second[0]);
		desired_point = second[0];
		//printf("%d \n", i);
		for (int j = 0; j < N_second; j++) {
			float distance = glm::distance(first[i], second[j]);
			if (distance < min_distance) {
				desired_point = second[j];
				min_distance = distance;
			}

		}
		corr[i] = desired_point;
	}
	//printf("out /n");
}

void find_mean(int n, glm::vec3 *idata, glm::vec3 host_mean) {

	for (int i = 0; i < n; i++) {

		host_mean += idata[i];
	}

	host_mean /= n;

}

void scanmatch::run_CPU(int N_first, int N_second, glm::vec3* first_points, glm::vec3* second_points) {
	
	
	//printf("here /n");
	glm::vec3* host_corr = (glm::vec3*)malloc(N_first * sizeof(glm::vec3));
	//printf("here /n");

	timer = clock();

	findmatch_cpu(N_first, N_second, first_points, second_points, host_corr);
	
	
	
	glm::vec3 host_mean_first(0.0f, 0.0f, 0.0f);
	glm::vec3 host_mean_corr(0.0f, 0.0f, 0.0f);

	for (int i = 0; i <N_first; i++) {

		host_mean_first += first_points[i];
		host_mean_corr += host_corr[i];
	}

	host_mean_first /= N_first;
	host_mean_corr /= N_first;
	

	//glm::mat3 W;
	float W[3][3] = { 0 };
	
	for (int k = 0; k < N_first; k++) {
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				W[i][j] += ((host_corr[k] - host_mean_corr)[i]) * ((first_points[k] - host_mean_first)[j]);
			}
		}
	}
	

	
	//glm::mat3 U;
	//glm::mat3 U = new glm::mat3[1];
	//glm::mat3 S;
	//glm::mat3 V;

	float U[3][3] = { 0 };
	float S[3][3] = { 0 };
	float V[3][3] = { 0 };
	//printf("%f \n", V[3][3]);
	svd(W[0][0], W[0][1], W[0][2], W[1][0], W[1][1], W[1][2], W[2][0], W[2][1], W[2][2],
		U[0][0], U[0][1], U[0][2], U[1][0], U[1][1], U[1][2], U[2][0], U[2][1], U[2][2],
		S[0][0], S[0][1], S[0][2], S[1][0], S[1][1], S[1][2], S[2][0], S[2][1], S[2][2],
		V[0][0], V[0][1], V[0][2], V[1][0], V[1][1], V[1][2], V[2][0], V[2][1], V[2][2]);

	//printf("%f \n", V[3][3]);

	glm::mat3 host_U =  glm::mat3(glm::vec3(U[0][0], U[1][0], U[2][0]), glm::vec3(U[0][1], U[1][1], U[2][1]), glm::vec3(U[0][2], U[1][2], U[2][2]));
	
	glm::mat3 host_rot = host_U * glm::mat3(glm::vec3(V[0][0], V[0][1], V[0][2]), glm::vec3(V[1][0], V[1][1], V[1][2]), glm::vec3(V[2][0], V[2][1], V[2][2]));


	glm::vec3 host_trans = host_mean_corr - (host_rot * host_mean_first);


	for (int i = 0; i < N_first; i++) {
		first_points[i] = (host_rot * first_points[i]) + host_trans;
	}

	timer = clock() - timer;
	time_per_iter = ((double)timer) / CLOCKS_PER_SEC;

	cudaMemcpy(dev_pos, first_points, N_first * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("copyDev_pos failed!");
	

	printf("(Time per Iter : %f \n", time_per_iter);
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/


/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/


void scanmatch::endSimulation() {
	cudaFree(dev_pos);
	cudaFree(dev_color);
	cudaFree(dev_first);
	cudaFree(dev_second);
	cudaFree(dev_corr);
}
