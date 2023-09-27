#ifndef NETPERFORMANCE_H
#define NETPERFORMANCE_H

#include "UnitTest.h"
#include "machine_learning.h"
#include "linear_algebra.h"

#define ONE_OR_ZERO(__var__) (((__var__) >= 0.5f) ? 1 : 0)

namespace NetTest {
	class Performance {
	public:

		template <typename T>
		static T XOR(T a, T b) {
			return ((T)(ONE_OR_ZERO(a) xor ONE_OR_ZERO(b)));
		}

		static LinearAlgebra::Matrix XOR(LinearAlgebra::Matrix& a);

		static MachineLearning::TrainingDataset xor_dataset();

		static void learn_xor();

		static void execute_all_tests();
	};
}

#endif // NETPERFORMANCE_H