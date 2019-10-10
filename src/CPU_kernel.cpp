#include "CPU_kernel.h"
#include "device_launch_parameters.h"
#include "glm/glm.hpp"
#include "svd3.h"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>




//glm::vec3* dev_first;
//glm::vec3* dev_second;



/*
void scanmatch::CPU::initSimulation(int N_first, int N_second, glm::vec3* first, glm::vec3* second) {
	cudaMalloc((void**)&dev_first, N_first * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_first failed!");
	cudaMemcpy(dev_first, first, sizeof(glm::vec3) * N_first, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_second, N_second * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_second failed!");
	cudaMemcpy(dev_second, second, sizeof(glm::vec3) * N_second, cudaMemcpyHostToDevice);

}

*/
void scanmatch::CPU::run(int N_first, int N_second, glm::vec3* first_points, glm::vec3* second_points) {

	glm::vec3* host_corr = (glm::vec3*)malloc(N_first * sizeof(glm::vec3*));

	findmatch(N_first, N_second, first_points, second_points, host_corr);
	
	glm::vec3 host_mean_first(0.0f, 0.0f, 0.0f);
	glm::vec3 host_mean_corr(0.0f, 0.0f, 0.0f);
	//glm::vec3 *mean = new glm::vec3[1];

	glm::vec3* host_centered_first = (glm::vec3*)malloc(N_first * sizeof(glm::vec3*));

	//*host_centered_first = *first_points;
	glm::vec3* host_centered_corr = (glm::vec3*)malloc(N_first * sizeof(glm::vec3*));
	//*host_centered_corr = host_corr;
	
	find_mean_and_sub(N_first, first_points, host_mean_first, host_centered_first);
	find_mean_and_sub(N_first, host_corr, host_mean_corr, host_centered_corr);

	glm::mat3 W;

	multiply_transpose(N_first, host_centered_first, host_centered_corr, W);

	glm::mat3 U;
	//glm::mat3 U = new glm::mat3[1];
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

	glm::vec3* new_buf = (glm::vec3*)malloc(N_first * sizeof(glm::vec3*));

	cudaMemcpy(new_buf, first_points, N_first * sizeof(glm::vec3), cudaMemcpyHostToHost);
	
	update(N_first, first_points, host_rot, host_trans, new_buf);

	cudaMemcpy(first_points, new_buf, N_first * sizeof(glm::vec3), cudaMemcpyHostToHost);
}

/*
void scanmatch::CPU::host_to_dev(int N_first, int N_second, glm::vec3* first_points, glm::vec3* second_points) {

	cudaMemcpy(dev_first, first_points, N_first * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_second, second_points, N_first * sizeof(glm::vec3), cudaMemcpyHostToDevice);
}
*/