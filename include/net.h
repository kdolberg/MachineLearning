#ifndef NET_H
#define NET_H

#include <list>
#include <utility>
#include <iostream>
#include "types.h"
#include "layer.h"
#include "activation_function.h"

#define LEARNING_RATE 0.1

namespace NetTest {
	class PrivateAPI;
}

namespace MachineLearning {

	typedef std::list<MachineLearning::LayerParams> Gradient;
	typedef std::vector<MachineLearning::uint> NetDef;

	/**
	 * @brief Calculates the error function
	 * @param net_output_data Output datset from the neural net for the current dataset and current parameters
	 * @param dataset_y_data Y-data from the training dataset
	 * @return Matrix representing the error vector (column) for each datapoint
	 */
	LinearAlgebra::Matrix error		(	const LinearAlgebra::Matrix& net_output_data,
										const LinearAlgebra::Matrix& dataset_y_data		);

	scalar_t error_avg				(	const LinearAlgebra::Matrix& net_output_data,
										const LinearAlgebra::Matrix& dataset_y_data		);

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



	class Net : public std::list<LayerParams> {
		friend NetTest::PrivateAPI;
	protected:
		std::list<LinearAlgebra::Matrix> pre_act_func_output;
		std::list<LinearAlgebra::Matrix> post_act_func_output;
		std::list<LayerParams> partial_derivatives;
		MachineLearning::ActivationFunction af;
	public:
		TrainingDataset td;
		LinearAlgebra::scalar_t learning_rate;
		Net() {
			// Set the learning rate to the default rate.
			this->learning_rate = LEARNING_RATE;
			this->af = MachineLearning::get_leaky_ReLU();
		}
		Net(NetDef def) : Net() {
			for (uint i = 0; i < def.size()-1; ++i) {
				uint num_inputs = def[i];
				uint num_outputs = def[i+1];
				LayerParams tmp(num_inputs,num_outputs);
				this->push_back(tmp);
			}
		}
		Net(NetDef def, bool rand) : Net(def) {
			if(rand) {
				for (Net::iterator i = this->begin(); i != this->end(); ++i) {
					i->randomize();
				}
			}
		}
		Net(NetDef def, scalar_t s) : Net(def) {
			for (Net::iterator i = this->begin(); i != this->end(); ++i) {
				i->weights.set_contents(s);
				i->biases.set_contents(s);
			}
		}
		std::string str() const {
			std::stringstream ss;
			for (Net::const_iterator i = this->cbegin(); i != this->cend(); ++i) {
				ss << (*i) << std::endl;
			}
			return ss.str();
		}
		void load_training_data(const TrainingDataset& td) {
			this->clear_data_caches();
			this->td = td;
		}
		Gradient calculate_gradient();
		LinearAlgebra::scalar_t error() const;
		uint get_num_data_points() const;
		uint get_num_outputs() const;
		uint get_num_inputs() const;
	protected:
		void forward_propagate();
		void backward_propagate();
		LinearAlgebra::Matrix get_last_output() const;
		void clear_data_caches();
		LinearAlgebra::Matrix error_ddx() const;
	}; // Net

} // MachineLearning

MachineLearning::Net& operator+=(MachineLearning::Net& a, MachineLearning::Gradient& b);

MachineLearning::Gradient operator-(const MachineLearning::Gradient& A, const MachineLearning::Gradient& B);

std::ostream& operator<<(std::ostream& os,const MachineLearning::Net& n);

std::ostream& operator<<(std::ostream& os, const MachineLearning::Gradient g);

// MachineLearning::NetGradIter& operator++(MachineLearning::NetGradIter& ni);

#endif //NET_H