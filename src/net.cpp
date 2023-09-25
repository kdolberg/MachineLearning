#include "net.h"

#include "UnitTest.h" // delete this line later

// const LinearAlgebra::Matrix& get_final_output_data(const std::list<MachineLearning::Layer>& layers) {
// 	return layers.back().get_post_act_func_output();
// }

LinearAlgebra::Matrix MachineLearning::error(const LinearAlgebra::Matrix& net_output_data,const LinearAlgebra::Matrix& dataset_y_data) {
	return LinearAlgebra::hadamard_square(dataset_y_data-net_output_data)*static_cast<LinearAlgebra::scalar_t>(0.5f);
}
LinearAlgebra::Matrix MachineLearning::error_ddx(const LinearAlgebra::Matrix& net_output_data, const LinearAlgebra::Matrix& dataset_y_data) {
	return (net_output_data-dataset_y_data);
}

MachineLearning::scalar_t MachineLearning::error_avg(const LinearAlgebra::Matrix& net_output_data, const LinearAlgebra::Matrix& dataset_y_data) {
	LinearAlgebra::Matrix E = MachineLearning::error(net_output_data,dataset_y_data);
	MachineLearning::uint N = 0;
	scalar_t sum = 0.0f;
	for (LinearAlgebra::mindex_t i = {0,0}; i.row < E.get_num_rows(); ++i.row) {
		for (i.col = 0; i.col < E.get_num_cols(); ++i.col) {
			sum += E[i];
			++N;
		}
	}
	return (sum/(1.0f*N));
}

LinearAlgebra::Matrix MachineLearning::Net::get_last_output() const {
	return this->post_act_func_output.back();
}

MachineLearning::scalar_t MachineLearning::Net::error() const {
	CONFIRM(!(this->get_last_output().empty()))
	CONFIRM(this->get_last_output().size()==this->td.y.size());
	return MachineLearning::error_avg(this->get_last_output(),this->td.y);
}

LinearAlgebra::Matrix MachineLearning::Net::error_ddx() const {
	CONFIRM(!(this->get_last_output().empty()));
	CONFIRM(!(this->td.y.empty()));
	return MachineLearning::error_ddx(this->get_last_output(),this->td.y);
}

void MachineLearning::Net::clear_data_caches() {
	this->pre_act_func_output.clear();
	this->post_act_func_output.clear();
	this->partial_derivatives.clear();
}

MachineLearning::Gradient MachineLearning::Net::calculate_gradient() {
	CONFIRM(!(this->td.x.empty()));
	CONFIRM(!(this->td.y.empty()));
	this->clear_data_caches();
	this->forward_propagate();
	this->backward_propagate();
	return this->partial_derivatives;
}

void MachineLearning::Net::forward_propagate() {
	//Empty all datacaches since we're starting from scratch
	this->clear_data_caches();

	//Add the input to the post_act_func_output
	this->post_act_func_output.push_back(this->td.x);
	for (auto i = this->begin(); i != this->end(); ++i) {
		this->pre_act_func_output.push_back(i->operator()(this->post_act_func_output.back()));
		this->post_act_func_output.push_back(this->af(this->pre_act_func_output.back()));
	}
}

MachineLearning::uint MachineLearning::Net::get_num_outputs() const {
	return this->back().get_num_outputs();
}

MachineLearning::uint MachineLearning::Net::get_num_inputs() const {
	return this->front().get_num_inputs();
}

MachineLearning::uint MachineLearning::Net::get_num_data_points() const {
	CONFIRM(!(this->td.y.empty()));
	CONFIRM(!(this->td.x.empty()));
	CONFIRM(this->td.x.get_num_cols()==this->td.y.get_num_cols());
	return this->td.x.get_num_cols();
}

typedef LinearAlgebra::ref_mindex ref_mindex;

void MachineLearning::Net::backward_propagate() {
	CONFIRM(!(this->td.y.empty()));
	CONFIRM(!(this->td.x.empty()));
	CONFIRM(!(this->pre_act_func_output.empty()));
	CONFIRM(!(this->post_act_func_output.empty()));
	CONFIRM(this->partial_derivatives.empty());

	// Calculate the error
	LinearAlgebra::Matrix dEdy = this->error_ddx();

	// Iterators
	auto pre_act_func_output = this->pre_act_func_output.crbegin();
	auto post_act_func_output = this->post_act_func_output.crbegin();

	// Indeces
	const uint bias_column_index = 0; // This is always zero.
	
	// Mindex pointers

	for (auto layer_iter = this->crbegin(); layer_iter != this->crend(); ++layer_iter, ++pre_act_func_output, ++post_act_func_output) {

		LinearAlgebra::Matrix derivatives_for_next_layer(MINDEX(layer_iter->get_num_inputs(),this->get_num_data_points()));
		LayerParams curr_partials(layer_iter->get_num_inputs(),layer_iter->get_num_outputs());
		LinearAlgebra::Matrix dfdx = this->af.ddx(*pre_act_func_output);
		LinearAlgebra::Matrix dfdx_dEdy = LinearAlgebra::hadamard_product(dfdx,dEdy);
		LinearAlgebra::Matrix curr_input = *(std::next(post_act_func_output));

		for (uint data_index = 0; data_index < this->get_num_data_points(); ++data_index) {

			for (uint output_index = 0; output_index < layer_iter->get_num_outputs(); ++output_index) {
				ref_mindex dfdx_dEdy_m	=	{	output_index	,	data_index			};
				ref_mindex biases_m		=	{	output_index	,	bias_column_index	};

				curr_partials.biases[biases_m] += dfdx_dEdy[dfdx_dEdy_m];

				for	(uint input_index = 0; input_index < layer_iter->get_num_inputs(); ++input_index) {
					ref_mindex weights_m					=	{	output_index	,	input_index		};
					ref_mindex curr_input_m					=	{	input_index		,	data_index		};
					ref_mindex derivatives_for_next_layer_m	=	{	input_index		,	data_index		};

					derivatives_for_next_layer[derivatives_for_next_layer_m] += layer_iter->weights[weights_m]*dfdx_dEdy[dfdx_dEdy_m];
					curr_partials.weights[weights_m] += curr_input[curr_input_m]*dfdx_dEdy[dfdx_dEdy_m];

				}

			}
		}
		dEdy = derivatives_for_next_layer;
		this->partial_derivatives.push_front(curr_partials/this->get_num_data_points());
	}
}

// /**
//  * @brief Computetes forward propagation for one layer
//  */
// void MachineLearning::forprop(std::list<MachineLearning::Layer>::iterator i,std::list<MachineLearning::Layer>::iterator end,const LinearAlgebra::Matrix& x_data) {
// 	if(i!=end) {
// 		MachineLearning::forprop(std::next(i,1),end,i->update_forprop_data_cache(x_data));
// 	}
// }

// /**
//  * @brief Performs forward propagation
//  * @param layers List of all layers in this net
//  * @param x_data Input data from the training dataset
//  */
// void MachineLearning::forprop(std::list<MachineLearning::Layer>& layers,const LinearAlgebra::Matrix& x_data) {
// 	MachineLearning::forprop(layers.begin(),layers.end(),x_data);
// }

// /**
//  * @brief Peforms backpropagation
//  * @param layers List of all layers in this net
//  * @param y_data Output data from the training dataset
//  */
// void MachineLearning::backprop(std::list<MachineLearning::Layer>& layers,const LinearAlgebra::Matrix& y_data) {
//     LinearAlgebra::Matrix from_prev_layer = MachineLearning::error_ddx(get_final_output_data(layers),y_data);
//     for (std::list<MachineLearning::Layer>::iterator i = layers.begin(); i != layers.end(); ++i) {
//         from_prev_layer = i->update_backprop_data_cache(from_prev_layer);
//     }
// }

// MachineLearning::NetGradIter& operator++(MachineLearning::NetGradIter& ni) {
// 	++ni.first;
// 	++ni.second;
// 	return ni;
// }

// MachineLearning::Net& operator+=(MachineLearning::Net& a, MachineLearning::Gradient& b) {
// 	assert(a.size()==b.size());
// 	for (MachineLearning::NetGradIter i(a.begin(),b.cbegin()); (i.first != a.end()) && (i.second != b.cend()); ++i) {
// 		*(i.first) += *(i.second);
// 	}
// 	return a;
// }
std::ostream& operator<<(std::ostream& os,const MachineLearning::Net& n) {
	os << n.str();
	return os;
}

std::ostream& operator<<(std::ostream& os, const MachineLearning::Gradient g) {
	for (MachineLearning::Gradient::const_iterator i = g.cbegin(); i != g.cend(); ++i) {
		os << (*i) << std::endl;
	}
	return os;
}

MachineLearning::Gradient operator-(const MachineLearning::Gradient& A, const MachineLearning::Gradient& B) {
	assert(A.size()==B.size());
	MachineLearning::Gradient ret;
	auto a = A.begin();
	auto b = B.begin();
	while (a != A.end()) {
		ret.push_back((*a)-(*b));
		++a;
		++b;
	}
	return ret;
}

// MachineLearning::Gradient MachineLearning::Net::get_gradient() const {
// 	MachineLearning::Gradient ret;
// 	return ret;
// }

// #define LEARNING_RATE 0.1

// void MachineLearning::Net::train(const TrainingDataset& data) {
// 	forprop(*this,data.x);
// 	backprop(*this,data.y);
// 	MachineLearning::Net n((std::vector<uint>){5,4,4,1});
// 	MachineLearning::Gradient g;
// 	// n += LEARNING_RATE*this->get_gradient();
// }

// void MachineLearning::Net::train(const TrainingDataset& data,int iters) {
// 	for (int i = 0; i < iters; ++i) {
// 		this->train(data);
// 	}
// }