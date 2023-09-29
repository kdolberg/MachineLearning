#include <iostream>
#include "UnitTest.h" // delete this line later
#include "net.h"

#define DELINEATOR '\n'

LinearAlgebra::Matrix MachineLearning::error(const LinearAlgebra::Matrix& net_output_data,const LinearAlgebra::Matrix& dataset_y_data) {
	return LinearAlgebra::hadamard_square(net_output_data- dataset_y_data)*static_cast<scalar>(0.5f);
}
LinearAlgebra::Matrix MachineLearning::error_ddx(const LinearAlgebra::Matrix& net_output_data, const LinearAlgebra::Matrix& dataset_y_data) {
	return (net_output_data-dataset_y_data);
}

MachineLearning::scalar MachineLearning::error_avg(const LinearAlgebra::Matrix& net_output_data, const LinearAlgebra::Matrix& dataset_y_data) {
	LinearAlgebra::Matrix E = MachineLearning::error(net_output_data,dataset_y_data);
	MachineLearning::uint N = 0;
	scalar sum = 0.0f;
	for (MachineLearning::mindex i = {0,0}; i.row < E.get_num_rows(); ++i.row) {
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

MachineLearning::scalar MachineLearning::Net::error() const {
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
	MachineLearning::Gradient ret = (this->partial_derivatives);
	ret *= ((-1.0f)*(this->learning_rate));
	return ret;
}

LinearAlgebra::Matrix MachineLearning::Net::operator()(const LinearAlgebra::Matrix& x_in) const {
	LinearAlgebra::Matrix ret = x_in;
	auto af = this->afs.cbegin();
	for (auto i = this->cbegin(); i != this->cend(); ++i) {
		ret = af->operator()(i->operator()(ret));
	}
	return ret;
}
LinearAlgebra::Matrix MachineLearning::Net::operator()() const {
	return this->operator()(this->td.x);
}

void MachineLearning::Net::forward_propagate() {
	//Empty all datacaches since we're starting from scratch
	this->clear_data_caches();

	//Add the input to the post_act_func_output
	this->post_act_func_output.push_back(this->td.x);
	auto af = this->afs.cbegin();
	for (auto i = this->begin(); i != this->end(); ++i,++af) {
		this->pre_act_func_output.push_back(i->operator()(this->post_act_func_output.back()));
		this->post_act_func_output.push_back(af->operator()(this->pre_act_func_output.back()));
	}
}

/**
 * @brief 
 */
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

void MachineLearning::Net::backward_propagate() {
	CONFIRM(!(this->td.y.empty()));
	CONFIRM(!(this->td.x.empty()));
	CONFIRM(!(this->pre_act_func_output.empty()));
	CONFIRM(!(this->post_act_func_output.empty()));
	CONFIRM(this->partial_derivatives.empty());

	// Calculate the derivative of the error function with respect to the output from forward propagation
	LinearAlgebra::Matrix dEdy = this->error_ddx();

	const uint bias_column_index = 0; // The biases of a layer are a single column. The index of that column is always 0.

	// Iterators to be used in the backpropagation for loop
	auto pre_act_func_output = this->pre_act_func_output.crbegin();
	auto post_act_func_output = this->post_act_func_output.crbegin();
	auto layer_iter = this->crbegin();
	auto af = this->afs.crbegin();

	// Backpropagation loop
	for (; layer_iter != this->crend(); ++layer_iter, ++pre_act_func_output, ++post_act_func_output, ++af) {

		LinearAlgebra::Matrix derivatives_for_next_layer(MINDEX(layer_iter->get_num_inputs(),this->get_num_data_points()));
		LayerParams curr_partials(layer_iter->get_num_inputs(),layer_iter->get_num_outputs());
		LinearAlgebra::Matrix dfdx = af->ddx(*pre_act_func_output);
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

std::string MachineLearning::Net::str() const {
	std::stringstream ss;
	for (Net::const_iterator i = this->cbegin(); i != this->cend(); ++i) {
		ss << (*i) << std::endl;
	}
	return ss.str();
}
void MachineLearning::Net::load_training_data(const MachineLearning::TrainingDataset& td) {
	this->clear_data_caches();
	this->td = td;
}

const MachineLearning::TrainingDataset& MachineLearning::Net::get_training_data() const {
	return this->td;
}

MachineLearning::scalar max(const MachineLearning::Gradient& n) {
	auto i = n.cbegin();
	MachineLearning::scalar ret = max(*i);
	++i;
	for (; i != n.cend(); ++i) {
		ret = std::max(ret,max(*i));
	}
	return ret;
}

MachineLearning::scalar min(const MachineLearning::Gradient& n) {
	auto i = n.cbegin();
	MachineLearning::scalar ret = min(*i);
	++i;
	for (; i != n.cend(); ++i) {
		ret = std::min(ret,min(*i));
	}
	return ret;
}

MachineLearning::scalar MachineLearning::Net::learn(const TrainingDataset& td) {
	this->load_training_data(td);
	MachineLearning::Gradient g = this->calculate_gradient();
	(*this) += g;
	return this->error();
}
MachineLearning::scalar MachineLearning::Net::learn() {
	return this->learn(this->get_training_data());
}

const MachineLearning::Gradient& MachineLearning::Net::get_partial_derivatives() const {
	assert(!this->partial_derivatives.empty());
	return this->partial_derivatives;
}

MachineLearning::Gradient& operator*=(MachineLearning::Gradient& g,MachineLearning::scalar s) {
	for (auto i = g.begin(); i != g.end(); ++i) {
		(*i) *= s;
	}
	return g;
}

MachineLearning::Net& MachineLearning::Net::operator+=(const MachineLearning::Gradient& g) {
	auto i = this->begin();
	auto j = g.cbegin();
	for (; i != this->end(); ++i,++j) {
		(*i) += (*j);
	}
	return (*this);
}

std::ostream& operator<<(std::ostream& os, const std::list<MachineLearning::ActivationFunction>& afs) {
	for (auto i = afs.cbegin(); i != afs.cend(); ++i) {
		os << (*i) << " ";
	}
	return os;
}

bool MachineLearning::Net::save(const char * filename) {
	std::ofstream file(filename, (std::ios::out | std::ios::binary));
	if (file.is_open()) {
		file << (*this);
		return true;
	} else {
		return false;
	}
}
MachineLearning::Net MachineLearning::Net::load(const char * filename){
	MachineLearning::Net ret;
	std::ifstream file(std::string(filename),std::ios::in | std::ios::binary);
	file >> ret;
	return ret;
}

// template <typename T>
// std::ofstream& save_gradient(std::ofstream& f, const T& g) {
// 	MachineLearning::uint num_layers = g.size();
// 	f << num_layers;
// 	// Send the number of inputs for each layer
// 	for (auto i = g.cbegin(); i != g.cend(); ++i) {
// 		f << i->get_num_inputs();
// 	}
// 	f << g.back().get_num_outputs();
// 	return f;
// }

// std::ofstream& operator<<(std::ofstream& f, const MachineLearning::Gradient& g) {
// 	return save_gradient<MachineLearning::Gradient>(f,g);
// }

// #define SIGMOID ((MachineLearning::uint)0)
// #define RELU ((MachineLearning::uint)1)

// std::ofstream& operator<<(std::ofstream& f, const MachineLearning::Net& n) {
// 	save_gradient<MachineLearning::Net>(f,n);
// 	f << DELINEATOR;

// 	for (auto i = n.afs.cbegin(); i != n.afs.cend(); ++i) {
// 		if (i->str() == std::string("sigmoid")) {
// 			f << SIGMOID;
// 		} else if (i->str() == std::string("leaky_ReLU")) {
// 			f << RELU;
// 		}
// 	}
// 	f << DELINEATOR;
// }

// std::ifstream& operator>>(std::ifstream& ifs, MachineLearning::NetDef& def) {
// 	assert(def.empty());
// 	MachineLearning::uint num_layers,curr;
// 	ifs >> num_layers;
// 	for (MachineLearning::uint i = 0; i < num_layers+1; ++i) {
// 		ifs >> curr;
// 		def.push_back(curr);
// 	}
// 	return ifs;
// }

std::ifstream& operator>>(std::ifstream& ifs, MachineLearning::Net& n) {
	return ifs;
}