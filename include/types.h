#ifndef TYPES_H
#define TYPES_H

#include "linear_algebra.h"
#include "activation_function.h"

#define UNUSED(var) {(void)(var);}

namespace MachineLearning {
	typedef LinearAlgebra::uint uint;

	typedef struct {
		LinearAlgebra::Matrix x;
		LinearAlgebra::Matrix y;
	} TrainingDataset;

} //MachineLearning

#endif //TYPES_H