#include "layer.h"

// LinearAlgebra::Matrix MachineLearning::calc_derivatives_to_pass_on(const LinearAlgebra::Matrix& weights,const LinearAlgebra::Matrix& naive_derivatives,const LinearAlgebra::Matrix& from_prev_layer) {
// 	LinearAlgebra::Matrix ret = LinearAlgebra::transpose(naive_derivatives)*from_prev_layer;
// 	return ret;
// }

// MachineLearning::LayerParams MachineLearning::calc_partial_derivatives(const LinearAlgebra::Matrix& derivatives,const LinearAlgebra::Matrix& pre_activation_function_output, const LinearAlgebra::Matrix& from_prev_layer) {
// 	MachineLearning::LayerParams ret;
// 	return ret;
// }

std::ostream& operator<<(std::ostream& os,const MachineLearning::LayerParams& lp) {
	for (LinearAlgebra::mindex_t i = {0,0}; i.row < lp.get_num_outputs(); ++i.row) {
		os << "[ "; 
		for (i.col = 0; i.col < lp.get_num_inputs(); ++i.col) {
			os << lp.get_weights()[i] << " ";
		}
		os << "]\t[ " << lp.get_biases()[MINDEX(i.row,0)] << " ]\n";
	}
	return os;
}

MachineLearning::BackPropIter MachineLearning::above(MachineLearning::BackPropIter it){
	return (--it);
}
MachineLearning::BackPropIter MachineLearning::below(MachineLearning::BackPropIter it){
	return (++it);
}
MachineLearning::ForPropIter MachineLearning::above(MachineLearning::ForPropIter it){
	return (++it);
}
MachineLearning::ForPropIter MachineLearning::below(MachineLearning::ForPropIter it){
	return (--it);
}

// ForPropIter MachineLearning::next(ForPropIter it){
// 	return (++it);
// }
// BackPropIter MachineLearning::next(BackPropIter it){
// 	return (++it);
// }
// ForPropIter MachineLearning::prev(ForPropIter it){
// 	return (--it);
// }
// BackPropIter MachineLearning::prev(BackPropIter it){
// 	return (--it);
// }

bool operator==(const MachineLearning::LayerParams& a, const MachineLearning::LayerParams& b) {
	return (a.get_weights()==b.get_weights()) && (a.get_biases()==b.get_biases());
}

// bool operator==(const MachineLearning::LayerParams a, const MachineLearning::LayerParams b) {
// 	return (a.get_weights()==b.get_weights()) && (a.get_biases()==b.get_biases());
// }

MachineLearning::LayerParams operator-(const MachineLearning::LayerParams& a, const MachineLearning::LayerParams& b) {
	MachineLearning::LayerParams ret = {a.get_weights()-b.get_weights(),a.get_biases()-b.get_biases()};
	return ret;
}

MachineLearning::LayerParams operator/(const MachineLearning::LayerParams& lp, LinearAlgebra::uint u) {
	return lp*(1.0f/u);
}

MachineLearning::LayerParams operator*(const MachineLearning::LayerParams& lp, LinearAlgebra::scalar_t u) {
	MachineLearning::LayerParams ret = {lp.get_weights()*u,lp.get_biases()*u};
	return ret;
}