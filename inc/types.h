#ifndef TYPES_H
#define TYPES_H

#include "linear_algebra.h"

#define UNUSED(var) {(void)(var);}

#define PRINT_LOC(...) std::cerr __VA_OPT__(<< #__VA_ARGS__ << "=" << __VA_ARGS__ << " ") << __FILE__ << " " << __FUNCTION__ << " " << __LINE__ << std::endl;

namespace MachineLearning {
	typedef LinearAlgebra::uint uint;

	typedef LinearAlgebra::scalar_t scalar;

	typedef LinearAlgebra::mindex_t mindex;

	typedef LinearAlgebra::ref_mindex ref_mindex;

	typedef struct {
		LinearAlgebra::Matrix x;
		LinearAlgebra::Matrix y;
	} TrainingDataset;

} //MachineLearning

bool operator==(const MachineLearning::TrainingDataset& a, const MachineLearning::TrainingDataset& b);

#endif //TYPES_H