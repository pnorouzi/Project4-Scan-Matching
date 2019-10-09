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
	namespace GPU {

		void initSimulation(int N_first, int N_second,float *first,float *second);

		void run(int N_first, int N_second);

	}
   
}
