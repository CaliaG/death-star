#ifndef RANDOM_H
#define RANDOM_H

#include <curand_kernel.h>

#include <glm/glm.hpp>

using namespace glm;

__device__ vec3 random_in_unit_sphere(curandState* rand_state)
{
	vec3 p;
	do {
		p = 2.0f *  vec3(curand_uniform(rand_state),
						 curand_uniform(rand_state),
						 curand_uniform(rand_state)) - vec3(1,1,1);
	} while (p.x * p.x + p.y * p.y + p.z * p.z >= 1.0);

	return p;
}

#endif
