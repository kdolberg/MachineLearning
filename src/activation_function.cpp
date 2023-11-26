#include <math.h>
#include <sstream>
#include "activation_function.h"

#ifndef LEAKINESS
#define LEAKINESS 0.1
#endif //LEAKINESS

MachineLearning::uint MachineLearning::unit(MachineLearning::scalar x) {
	return x;
}

MachineLearning::ActivationFunction MachineLearning::sym2ActFunc(func_sym sym) {
	if(sym==SIGMOID)
		return get_sigmoid();
	if(sym==LEAKY_RELU)
		return get_leaky_ReLU();
	return ActivationFunction(((ActivationFunctionStruct){unit,unit,UNDEFINED}));
}

// ActivationFunction methods

MachineLearning::scalar MachineLearning::ActivationFunction::operator()(scalar in) const {
	return this->function(in);
}

MachineLearning::scalar MachineLearning::ActivationFunction::ddx(scalar in) const {
	return this->derivative(in);
}

std::string MachineLearning::sym2name(func_sym sym) {
	if(sym==SIGMOID) {
		return SIGMOID_NAME;
	}
	if(sym==LEAKY_RELU) {
		return LEAKY_RELU_NAME;
	}
	return UNDEFINED_NAME;
}

std::string MachineLearning::ActivationFunction::str() const {
	return MachineLearning::sym2name(this->sym);
}

MachineLearning::func_sym MachineLearning::ActivationFunction::get_sym() const {
	return (this->sym);
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
	MachineLearning::ActivationFunction ret((ActivationFunctionStruct){leaky_ReLU,leaky_ReLU_ddx,LEAKY_RELU});
	return ret;
}

MachineLearning::ActivationFunction MachineLearning::get_sigmoid() {
	MachineLearning::ActivationFunction ret((ActivationFunctionStruct){sigmoid,sigmoid_ddx,SIGMOID});
	return ret;
}

std::ostream& operator<<(std::ostream& os, const MachineLearning::ActivationFunction& af) {
	os << af.str();
	return os;
}

bool operator==(const MachineLearning::ActivationFunction& a, const MachineLearning::ActivationFunction& b) {
	return a.get_sym() == b.get_sym();
}