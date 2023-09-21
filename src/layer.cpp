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