#pragma once
#include <glm/glm.hpp>

namespace scanmatch {
	namespace GPU {

		inline void initSimulation(int N_first, int N_second, glm::vec3* first, glm::vec3* second);

		inline void run(int N_first, int N_second);

	}
   
}
