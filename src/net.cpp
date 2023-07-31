#include "net.h"

LinearAlgebra::Matrix MachineLearning::error(const LinearAlgebra::Matrix& net_output_data,const LinearAlgebra::Matrix& dataset_y_data) {
	return LinearAlgebra::hadamard_square(dataset_y_data-net_output_data)*0.5f;
}
LinearAlgebra::Matrix MachineLearning::error_ddx(const LinearAlgebra::Matrix& net_output_data, const LinearAlgebra::Matrix& dataset_y_data) {
	return (net_output_data-dataset_y_data);
}

ForDataCache forprop (std::list<Layer>::const_iterator i, std::list<Layer>::const_iterator end, const LinearAlgebra::Matrix& x_data) {
	
}

/**
 * @brief Performs forward propagation
 * @param layers List of all layers in this net
 * @param x_data Input data from the training dataset
 */
ForDataCache forprop (const std::list<Layer>& layers,const LinearAlgebra::Matrix& x_data) {
}

/**
 * @brief Peforms backpropagation
 * @param layers List of all layers in this net
 * @param y_data Output data from the training dataset
 */
BackDataCache backprop 			(	const std::list<Layer>& layers,
									const ForDataCache& for_data,
									const LinearAlgebra::Matrix& y_data				) {

}

std::ostream& operator<<(std::ostream& os,const MachineLearning::Net n) {
	os << n.str();
	return os;
}