#include <math.h>
#include <sstream>
#include "activation_function.h"

#ifndef LEAKINESS
#define LEAKINESS 0.1
#endif //LEAKINESS

// ActivationFunction methods

MachineLearning::scalar MachineLearning::ActivationFunction::operator()(scalar in) const {
	return this->function(in);
}

MachineLearning::scalar MachineLearning::ActivationFunction::ddx(scalar in) const {
	return this->derivative(in);
}

const std::string& MachineLearning::ActivationFunction::str() const {
	return this->name;
}

/**
 * @brief Calculates the leaky ReLU function for the input value x
 */
MachineLearning::scalar leaky_ReLU(MachineLearning::scalar x) {
	if(x>0) {
		return x;
	} else {
		return LEAKINESS*x;
	}
}

/**
 * @brief Calculates derivative of leaky ReLU function
 */
MachineLearning::scalar leaky_ReLU_ddx(MachineLearning::scalar x) {
	if(x>0) {
		return 1;
	} else {
		return LEAKINESS;
	}
}

/**
 * @brief Calculates sigmoid function
 */
MachineLearning::scalar sigmoid(MachineLearning::scalar x) {
	return 1.0/(1.0+exp(0-x));
}

/**
 * @brief Calculates derivative of sigmoid function
 */
MachineLearning::scalar sigmoid_ddx(MachineLearning::scalar x) {
	return sigmoid(x)*(1-sigmoid(x));
}

/**
 * @brief Builds and returns an ActivationFunction object that containing the leaky ReLU function
 */
MachineLearning::ActivationFunction MachineLearning::get_leaky_ReLU() {
	MachineLearning::ActivationFunction ret((ActivationFunctionStruct){leaky_ReLU,leaky_ReLU_ddx});
	ret.name = "leaky_ReLU";
	return ret;
}

MachineLearning::ActivationFunction MachineLearning::get_sigmoid() {
	MachineLearning::ActivationFunction ret((ActivationFunctionStruct){sigmoid,sigmoid_ddx});
	ret.name = "sigmoid";
	return ret;
}

std::ostream& operator<<(std::ostream& os, const MachineLearning::ActivationFunction& af) {
	os << af.str();
	return os;
}