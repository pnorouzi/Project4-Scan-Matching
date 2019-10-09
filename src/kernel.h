#pragma once

#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <cmath>
#include <vector>

namespace scanmatch {
		void copyToDevice(int N_first, int N_second, glm::vec3 first_points, glm::vec3 second_points);
		void initSimulation(int N_first, int N_second, glm::vec3 first_points, glm::vec3 second_points);
		void copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities);
		float* getDevPos();
		void endSimulation();
}
