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

		void initSimulation(int N_first, int N_second,float *first,float *second);
	    void findCorrespondence(float dt);
	    void stepSimulationScatteredGrid(float dt);
	    void stepSimulationCoherentGrid(float dt);
	    void copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities);

	    void endSimulation();
	    void unitTest();

	}
   
}
