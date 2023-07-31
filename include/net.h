#ifndef NET_H
#define NET_H

#include <list>
#include "types.h"
#include "activation_function.h"

namespace MachineLearning {

	/**
	 * @brief Calculates the error function
	 * @param net_output_data Output datset from the neural net for the current dataset and current parameters
	 * @param dataset_y_data Y-data from the training dataset
	 * @return Matrix representing the error vector (column) for each datapoint
	 */
	LinearAlgebra::Matrix error		(	const LinearAlgebra::Matrix& net_output_data,
										const LinearAlgebra::Matrix& dataset_y_data		);

	/**
	 * @brief Derivative of LinearAlgebra::Matrix MachineLearning::error (const LinearAlgebra::Matrix& net_output_data,const LinearAlgebra::Matrix& dataset_y_data)
	 * @param net_output_data Output datset from the neural net for the current dataset and current parameters
	 * @param dataset_y_data Y-data from the training dataset
	 * @return Matrix representing derivatives of the error vector (column) for each datapoint
	 */
	LinearAlgebra::Matrix error_ddx	(	const LinearAlgebra::Matrix& net_output_data,
										const LinearAlgebra::Matrix& dataset_y_data		);

	/**
	 * @brief Performs forward propagation
	 * @param layers List of all layers in this net
	 * @param x_data Input data from the training dataset
	 */
	ForDataCache forprop			(	const std::list<Layer>& layers,
										const LinearAlgebra::Matrix& x_data				);

	/**
	 * @brief Peforms backpropagation
	 * @param layers List of all layers in this net
	 * @param y_data Output data from the training dataset
	 */
	BackDataCache backprop 			(	const std::list<Layer>& layers,
										const ForDataCache& for_data,
										const LinearAlgebra::Matrix& y_data				);

	/**
	 * 
	 */
	class Net {
	protected:
		std::list<Layer> layers;
	public:
		Net() {}
		Net(std::vector<uint> def) : Net() {
			for (uint i = 0; i < def.size()-1; ++i) {
				uint num_inputs = def[i];
				uint num_outputs = def[i+1];
				Layer tmp(num_inputs,num_outputs,get_leaky_ReLU(),true);
				layers.push_back(tmp);
			}
		}
		std::list<Layer>::const_iterator layer_cbegin() const {
			return this->layers.cbegin();
		}
		std::list<Layer>::const_iterator layer_cend() const {
			return this->layers.cend();
		}
	public:
		std::string str() const {
			std::stringstream ss;
			for (std::list<Layer>::const_iterator i = this->layer_cbegin(); i != this->layer_cend(); ++i) {
				ss << i->parameters.weights << std::endl;
				ss << i->parameters.biases << std::endl;
			}
			return ss.str();
		}
	}; //Net
} //MachineLearning

std::ostream& operator<<			(	std::ostream& os,const MachineLearning::Net n)	;

#endif //NET_H