#include "layer.h"

void MachineLearning::LayerParams::randomize() {
	this->weights.randomize();
	this->biases.randomize();
}

const LinearAlgebra::Matrix& MachineLearning::LayerParams::get_weights() const {
	return this->weights;
}

const LinearAlgebra::Matrix& MachineLearning::LayerParams::get_biases() const {
	return this->biases;
}

LinearAlgebra::Matrix MachineLearning::LayerParams::operator()(const LinearAlgebra::Matrix& in_signal) const {
	return this->call_op<LinearAlgebra::Matrix>(in_signal);
}

LinearAlgebra::HorizontalVector MachineLearning::LayerParams::operator()(const LinearAlgebra::HorizontalVector& in_signal) const {
	return this->call_op<LinearAlgebra::HorizontalVector>(in_signal);
}

LinearAlgebra::VerticalVector MachineLearning::LayerParams::operator()(const LinearAlgebra::VerticalVector& in_signal) const {
	return this->call_op<LinearAlgebra::VerticalVector>(in_signal);
}

MachineLearning::uint MachineLearning::LayerParams::get_num_inputs() const {
	return this->get_weights().get_num_cols();
}
MachineLearning::uint MachineLearning::LayerParams::get_num_outputs() const {
	assert(this->get_weights().get_num_rows()==this->get_biases().get_num_rows());
	return this->get_weights().get_num_rows();
}

MachineLearning::LayerParams& MachineLearning::LayerParams::operator+=(const MachineLearning::LayerParams& b) {
	this->weights += b.weights;
	this->biases += b.biases;
	return (*this);
}
bool MachineLearning::LayerParams::operator==(const LayerParams& ls) const {
	return (this->get_weights()==ls.get_weights()) && (this->get_biases()==ls.get_biases());
}

std::ostream& operator<<(std::ostream& os,const MachineLearning::LayerParams& lp) {
	for (MachineLearning::mindex i = {0,0}; i.row < lp.get_num_outputs(); ++i.row) {
		os << "[ "; 
		for (i.col = 0; i.col < lp.get_num_inputs(); ++i.col) {
			os << lp.get_weights()[i] << " ";
		}
		os << "]\t[ " << lp.get_biases()[MINDEX(i.row,0)] << " ]\n";
	}
	return os;
}

bool operator==(const MachineLearning::LayerParams& a, const MachineLearning::LayerParams& b) {
	return (a.get_weights()==b.get_weights()) && (a.get_biases()==b.get_biases());
}

MachineLearning::LayerParams operator-(const MachineLearning::LayerParams& a, const MachineLearning::LayerParams& b) {
	MachineLearning::LayerParams ret = {a.get_weights()-b.get_weights(),a.get_biases()-b.get_biases()};
	return ret;
}

MachineLearning::LayerParams operator/(const MachineLearning::LayerParams& lp, LinearAlgebra::uint u) {
	return lp*(1.0f/u);
}

MachineLearning::LayerParams operator*(const MachineLearning::LayerParams& lp, MachineLearning::scalar u) {
	MachineLearning::LayerParams ret = {lp.get_weights()*u,lp.get_biases()*u};
	return ret;
}