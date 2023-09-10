#include "layer.h"

LinearAlgebra::Matrix MachineLearning::calc_derivatives_to_pass_on(const LinearAlgebra::Matrix& weights,const LinearAlgebra::Matrix& naive_derivatives,const LinearAlgebra::Matrix& from_prev_layer) {
	LinearAlgebra::Matrix ret = LinearAlgebra::transpose(naive_derivatives)*from_prev_layer;
	return ret;
}

MachineLearning::LayerParams MachineLearning::calc_partial_derivatives(const LinearAlgebra::Matrix& derivatives,const LinearAlgebra::Matrix& pre_activation_function_output, const LinearAlgebra::Matrix& from_prev_layer) {
	MachineLearning::LayerParams ret;
	return ret;
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