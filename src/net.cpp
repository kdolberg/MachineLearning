#include "net.h"

LinearAlgebra::Matrix MachineLearning::error(const LinearAlgebra::Matrix& net_output_data,const LinearAlgebra::Matrix& dataset_y_data) {
	return LinearAlgebra::hadamard_square(dataset_y_data-net_output_data)*0.5f;
}
LinearAlgebra::Matrix MachineLearning::error_ddx(const LinearAlgebra::Matrix& net_output_data, const LinearAlgebra::Matrix& dataset_y_data) {
	return (net_output_data-dataset_y_data);
}

/**
 * @brief Computetes forward propagation for one layer
 */
void MachineLearning::forprop(std::list<Layer>::iterator i,std::list<Layer>::iterator end,const LinearAlgebra::Matrix& x_data) {
	if(i!=end) {
		forprop(std::next(i,1),end,i->update_forprop_data_cache(x_data));
	}
}

/**
 * @brief Performs forward propagation
 * @param layers List of all layers in this net
 * @param x_data Input data from the training dataset
 * 
 * NOTE: There is probably a way better way to do this! What if we switched to having a class that includes both the layer parameters, the 
 * activation function, AND the data caches? That way, we don't have to do memory allocation with each iteration and this should speed things
 * up.
 * 
 * Perhaps it would be worth writing it both ways and comparing the results in terms of speed.
 */
void MachineLearning::forprop(std::list<Layer>& layers,const LinearAlgebra::Matrix& x_data) {
	MachineLearning::forprop(layers.begin(),layers.end(),x_data);
}

/**
 * @brief Peforms backpropagation
 * @param layers List of all layers in this net
 * @param y_data Output data from the training dataset
 */
void MachineLearning::backprop (	const std::list<Layer>& layers,
									const LayerForDataCache& for_data,
									const LinearAlgebra::Matrix& y_data 	) {

}

std::ostream& operator<<(std::ostream& os,const MachineLearning::Net& n) {
	os << n.str();
	return os;
}