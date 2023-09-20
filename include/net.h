#ifndef NET_H
#define NET_H

#include <list>
#include <utility>
#include "types.h"
#include "layer.h"
#include "activation_function.h"

namespace MachineLearning {

	typedef std::list<MachineLearning::LayerParams> Gradient;

	// /**
	//  * @brief Calculates the error function
	//  * @param net_output_data Output datset from the neural net for the current dataset and current parameters
	//  * @param dataset_y_data Y-data from the training dataset
	//  * @return Matrix representing the error vector (column) for each datapoint
	//  */
	// LinearAlgebra::Matrix error		(	const LinearAlgebra::Matrix& net_output_data,
	// 									const LinearAlgebra::Matrix& dataset_y_data		);

	// /**
	//  * @brief Derivative of LinearAlgebra::Matrix MachineLearning::error (const LinearAlgebra::Matrix& net_output_data,const LinearAlgebra::Matrix& dataset_y_data)
	//  * @param net_output_data Output datset from the neural net for the current dataset and current parameters
	//  * @param dataset_y_data Y-data from the training dataset
	//  * @return Matrix representing derivatives of the error vector (column) for each datapoint
	//  */
	// LinearAlgebra::Matrix error_ddx	(	const LinearAlgebra::Matrix& net_output_data,
	// 									const LinearAlgebra::Matrix& dataset_y_data		);

	// /**
	//  * @brief Performs forward propagation
	//  * @param layers List of all layers in this net
	//  * @param x_data Input data from the training dataset
	//  */
	// void forprop					(	std::list<Layer>::iterator i,
	// 									std::list<Layer>::iterator end,
	// 									const LinearAlgebra::Matrix& x_data				);

	// void forprop					(	std::list<Layer>& layers,
	// 									const LinearAlgebra::Matrix& x_data				);
	// /**
	//  * @brief Peforms backpropagation
	//  * @param layers List of all layers in this net
	//  * @param y_data Output data from the training dataset
	//  */
	// void backprop					(	std::list<MachineLearning::Layer>& layers,
	// 									const LinearAlgebra::Matrix& y_data				);



	class Net : public std::list<LayerStruct> {
	public:
		using std::list<LayerStruct>::list;
		Net() : std::list<LayerStruct>::list() {}
		Net(std::vector<uint> def) : Net() {
			for (uint i = 0; i < def.size()-1; ++i) {
				uint num_inputs = def[i];
				uint num_outputs = def[i+1];
				LayerStruct tmp(num_inputs,num_outputs);
				this->push_back(tmp);
			}
		}
		Net(std::vector<uint> def, bool rand) : Net(def) {
			if(rand) {
				for (Net::iterator i = this->begin(); i != this->end(); ++i) {
					i->params.randomize();
				}
			}
		}
		// std::string str() const {
		// 	std::stringstream ss;
		// 	for (std::list<Layer>::const_iterator i = this->cbegin(); i != this->cend(); ++i) {
		// 		ss << (MachineLearning::LayerParams)(*i) << std::endl;
		// 	}
		// 	return ss.str();
		// }
		// void train(const TrainingDataset& data);
		// void train(const TrainingDataset& data, int iters);
		// Gradient get_gradient() const;
	}; // Net

} // MachineLearning

// MachineLearning::Net& operator+=(MachineLearning::Net& a, MachineLearning::Gradient& b);

std::ostream& operator<<(std::ostream& os,const MachineLearning::Net& n);

// MachineLearning::NetGradIter& operator++(MachineLearning::NetGradIter& ni);

#endif //NET_H