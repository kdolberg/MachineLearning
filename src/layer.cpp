#include "layer.h"

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