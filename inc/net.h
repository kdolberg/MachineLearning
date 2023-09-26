#ifndef NET_H
#define NET_H

#include <list>
#include <utility>
#include <iostream>
#include <algorithm>
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

	scalar error_avg				(	const LinearAlgebra::Matrix& net_output_data,
										const LinearAlgebra::Matrix& dataset_y_data		);

	/**
	 * @brief Derivative of LinearAlgebra::Matrix MachineLearning::error (const LinearAlgebra::Matrix& net_output_data,const LinearAlgebra::Matrix& dataset_y_data)
	 * @param net_output_data Output datset from the neural net for the current dataset and current parameters
	 * @param dataset_y_data Y-data from the training dataset
	 * @return Matrix representing derivatives of the error vector (column) for each datapoint
	 */
	LinearAlgebra::Matrix error_ddx	(	const LinearAlgebra::Matrix& net_output_data,
										const LinearAlgebra::Matrix& dataset_y_data		);

	class Net : public std::list<LayerParams> {
		friend NetTest::PrivateAPI;
	protected:
		std::list<LinearAlgebra::Matrix> pre_act_func_output;
		std::list<LinearAlgebra::Matrix> post_act_func_output;
		std::list<LayerParams> partial_derivatives;
		TrainingDataset td;
	public:
		MachineLearning::ActivationFunction af;
		scalar learning_rate;
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
		Net(NetDef def, scalar s) : Net(def) {
			for (Net::iterator i = this->begin(); i != this->end(); ++i) {
				i->weights.set_contents(s);
				i->biases.set_contents(s);
			}
		}
		std::string str() const;
		void load_training_data(const MachineLearning::TrainingDataset& td);
		const TrainingDataset& get_training_data() const;
		Gradient calculate_gradient();
		scalar error() const;
		uint get_num_data_points() const;
		uint get_num_outputs() const;
		uint get_num_inputs() const;
		LinearAlgebra::Matrix operator()(const LinearAlgebra::Matrix&) const;
		LinearAlgebra::Matrix operator()() const;
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

MachineLearning::scalar max(const MachineLearning::Gradient& n);

MachineLearning::scalar min(const MachineLearning::Gradient& n);

#endif //NET_H