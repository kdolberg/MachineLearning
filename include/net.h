#ifndef NET_H
#define NET_H

#include <list>
#include <utility>
#include "types.h"
#include "layer.h"
#include "activation_function.h"

#define LEARNING_RATE 0.1

class NetTest;

namespace MachineLearning {

	typedef std::list<MachineLearning::LayerParams> Gradient;
	typedef std::vector<MachineLearning::uint> NetDef;

	// /**
	//  * @brief Calculates the error function
	//  * @param net_output_data Output datset from the neural net for the current dataset and current parameters
	//  * @param dataset_y_data Y-data from the training dataset
	//  * @return Matrix representing the error vector (column) for each datapoint
	//  */
	// LinearAlgebra::Matrix error		(	const LinearAlgebra::Matrix& net_output_data,
	// 									const LinearAlgebra::Matrix& dataset_y_data		);

	/**
	 * @brief Derivative of LinearAlgebra::Matrix MachineLearning::error (const LinearAlgebra::Matrix& net_output_data,const LinearAlgebra::Matrix& dataset_y_data)
	 * @param net_output_data Output datset from the neural net for the current dataset and current parameters
	 * @param dataset_y_data Y-data from the training dataset
	 * @return Matrix representing derivatives of the error vector (column) for each datapoint
	 */
	LinearAlgebra::Matrix error_ddx	(	const LinearAlgebra::Matrix& net_output_data,
										const LinearAlgebra::Matrix& dataset_y_data		);

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
		friend NetTest;
	public:
		using std::list<LayerStruct>::list;
		Net() : std::list<LayerStruct>::list() {}
		Net(NetDef def) : Net() {
			for (uint i = 0; i < def.size()-1; ++i) {
				uint num_inputs = def[i];
				uint num_outputs = def[i+1];
				LayerStruct tmp(num_inputs,num_outputs);
				this->push_back(tmp);
			}
		}
		Net(NetDef def, bool rand) : Net(def) {
			if(rand) {
				for (Net::iterator i = this->begin(); i != this->end(); ++i) {
					i->params.randomize();
				}
			}
		}
		Net(NetDef def, scalar_t s) : Net(def) {
			for (Net::iterator i = this->begin(); i != this->end(); ++i) {
				i->params.weights.set_contents(s);
				i->params.biases.set_contents(s);
			}
		}
		std::string str() const {
			std::stringstream ss;
			for (Net::const_iterator i = this->cbegin(); i != this->cend(); ++i) {
				ss << i->params << std::endl;
			}
			return ss.str();
		}
		// void train(const TrainingDataset& data);
		// void train(const TrainingDataset& data, int iters);
		/**
		 * @brief
		 * @return
		 */
		BackPropIter rend() {
			return (this->std::list<LayerStruct>::rend());
		}
		BackPropIter rbegin() {
			return (this->std::list<LayerStruct>::rbegin());
		}
		ForPropIter end() {
			return (this->std::list<LayerStruct>::end());
		}
		ForPropIter begin() {
			return (this->std::list<LayerStruct>::begin());
		}
		LinearAlgebra::Matrix forward_propagate(const LinearAlgebra::Matrix& x_data) {
			for (ForPropIter fpi = this->begin(); fpi != this->end(); ++fpi) {
				if(fpi == this->begin()) {
					fpi.update_data_cache(x_data);
				} else {
					fpi.update_data_cache();
				}
			}
			return this->back().fordata.post_act_func_output;
		}
		void backward_propagate(const LinearAlgebra::Matrix& x_data,const LinearAlgebra::Matrix& dEdy) {
			for (BackPropIter bpi = this->rbegin(); bpi != this->rend(); ++bpi) {
				if (bpi==this->rbegin()) {
					bpi.update_data_cache_output_layer(dEdy);
				} else if (bpi==this->rend().above()) {
					bpi.update_data_cache_input_layer(x_data);
				} else {
					bpi.update_data_cache();
				}
			}
		}
		void learn(const TrainingDataset& data) {
			LinearAlgebra::Matrix net_y_out = this->forward_propagate(data.x);
			LinearAlgebra::Matrix dEdy = error_ddx(data.x,net_y_out);
			this->backward_propagate(data.x,dEdy);
		}
		Gradient extract_gradient() const {
			Gradient ret;
			for (auto i = this->cbegin(); i != this->cend(); ++i) {
				ret.push_back(i->backdata.partial_derivatives);
			}
			return ret;
		}
	}; // Net

} // MachineLearning

MachineLearning::Net& operator+=(MachineLearning::Net& a, MachineLearning::Gradient& b);

std::ostream& operator<<(std::ostream& os,const MachineLearning::Net& n);

// MachineLearning::NetGradIter& operator++(MachineLearning::NetGradIter& ni);

#endif //NET_H