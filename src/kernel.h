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
		void copyToDevice(int N1, int N2, float* xpoints, float* ypoints);
		void initSimulation(int N1, int N2, float* xpoints, float* ypoints);
		void copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities);
		float* getDevPos();
		void endSimulation();
}
