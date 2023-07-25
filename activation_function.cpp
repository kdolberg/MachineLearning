#include "activation_function.h"
#include "math.h"

// 1/(1+exp(-x));

#ifndef LEAKINESS
#define LEAKINESS 0.1
#endif //LEAKINESS

/**
 * @brief Calculates the leaky ReLU function for the input value x
 */
LinearAlgebra::scalar_t leaky_ReLU(LinearAlgebra::scalar_t x) {
	if(x>0) {
		return x;
	} else {
		return LEAKINESS*x;
	}
}

/**
 * @brief Calculates derivative of leaky ReLU function
 */
LinearAlgebra::scalar_t leaky_ReLU_ddx(LinearAlgebra::scalar_t x) {
	if(x>0) {
		return 1;
	} else {
		return LEAKINESS;
	}
}

/**
 * @brief Calculates sigmoid function
 */
LinearAlgebra::scalar_t sigmoid(LinearAlgebra::scalar_t x) {
	return 1.0/(1.0+exp(0-x));
}

/**
 * @brief Calculates derivative of sigmoid function
 */
LinearAlgebra::scalar_t sigmoid_ddx(LinearAlgebra::scalar_t x) {
	return sigmoid(x)*(1-sigmoid(x));
}

/**
 * @brief Builds and returns an ActivationFunction object that containing the leaky ReLU function
 */
MachineLearning::ActivationFunction MachineLearning::get_leaky_ReLU() {
	static MachineLearning::ActivationFunction ret((ActivationFunctionStruct){leaky_ReLU,leaky_ReLU_ddx});
	return ret;
}

MachineLearning::ActivationFunction MachineLearning::get_sigmoid() {
	static MachineLearning::ActivationFunction ret((ActivationFunctionStruct){sigmoid,sigmoid_ddx});
	return ret;
}