#include "layer.h"

LinearAlgebra::Matrix MachineLearning::calc_derivatives_to_pass_on(const LinearAlgebra::Matrix& derivatives,const LinearAlgebra::Matrix& weights) {;
	return derivatives;
}
LinearAlgebra::Matrix MachineLearning::calc_partial_derivatives(const LinearAlgebra::Matrix& derivatives,const LinearAlgebra::Matrix& pre_activation_function_output) {
	return derivatives;
}

MachineLearning::LayerParams& operator+=(MachineLearning::LayerParams& a, const MachineLearning::LayerParams& b) {
	a.weights += b.weights;
	a.biases += b.biases;
	return a;
}

MachineLearning::Layer& operator+=(MachineLearning::Layer& a, const MachineLearning::LayerParams& b) {
	a.parameters += b;
	return a;
}

MachineLearning::Layer& operator+=(MachineLearning::Layer& a, const MachineLearning::Layer& b) {
	a += b.parameters;
	return a;
}