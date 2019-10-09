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
	namespace CPU {

		void initSimulation(int N_first, int N_second, glm::vec3* first, glm::vec3* second);

		void run(int N_first, int N_second, glm::vec3* first_points, glm::vec3* second_points);

		void host_to_dev(int N_first, int N_second, glm::vec3* first_points, glm::vec3* second_points);
	}
   
}
