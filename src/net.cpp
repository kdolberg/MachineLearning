#include "net.h"

const LinearAlgebra::Matrix& get_final_output_data(const std::list<MachineLearning::Layer>& layers) {
	return layers.back().get_post_act_func_output();
}

LinearAlgebra::Matrix MachineLearning::error(const LinearAlgebra::Matrix& net_output_data,const LinearAlgebra::Matrix& dataset_y_data) {
	return LinearAlgebra::hadamard_square(dataset_y_data-net_output_data)*0.5f;
}
LinearAlgebra::Matrix MachineLearning::error_ddx(const LinearAlgebra::Matrix& net_output_data, const LinearAlgebra::Matrix& dataset_y_data) {
	return (net_output_data-dataset_y_data);
}

/**
 * @brief Computetes forward propagation for one layer
 */
void MachineLearning::forprop(std::list<MachineLearning::Layer>::iterator i,std::list<MachineLearning::Layer>::iterator end,const LinearAlgebra::Matrix& x_data) {
	if(i!=end) {
		forprop(std::next(i,1),end,i->update_forprop_data_cache(x_data));
	}
}

/**
 * @brief Performs forward propagation
 * @param layers List of all layers in this net
 * @param x_data Input data from the training dataset
 */
void MachineLearning::forprop(std::list<MachineLearning::Layer>& layers,const LinearAlgebra::Matrix& x_data) {
	MachineLearning::forprop(layers.begin(),layers.end(),x_data);
}

/**
 * @brief Peforms backpropagation
 * @param layers List of all layers in this net
 * @param y_data Output data from the training dataset
 */
void MachineLearning::backprop(std::list<MachineLearning::Layer>& layers,const LinearAlgebra::Matrix& y_data) {
    LinearAlgebra::Matrix from_prev_layer = MachineLearning::error_ddx(get_final_output_data(layers),y_data);
    for (std::list<MachineLearning::Layer>::iterator i = layers.begin(); i != layers.end(); ++i) {
        from_prev_layer = i->update_backprop_data_cache(from_prev_layer);
    }
}

MachineLearning::NetGradIter& operator++(MachineLearning::NetGradIter& ni) {
	++ni.first;
	++ni.second;
	return ni;
}

MachineLearning::Net& operator+=(MachineLearning::Net& a, MachineLearning::Gradient& b) {
	assert(a.size()==b.size());
	for (MachineLearning::NetGradIter i(a.begin(),b.cbegin()); (i.first != a.end()) && (i.second != b.cend()); ++i) {
		*(i.first) += *(i.second);
	}
	return a;
}

std::ostream& operator<<(std::ostream& os,const MachineLearning::Net& n) {
	os << n.str();
	return os;
}

MachineLearning::Gradient MachineLearning::Net::get_gradient() const {
	MachineLearning::Gradient ret;
	return ret;
}

#define LEARNING_RATE 0.1

void MachineLearning::Net::train(const TrainingDataset& data) {
	forprop(*this,data.x);
	backprop(*this,data.y);
	MachineLearning::Net n((std::vector<uint>){5,4,4,1});
	MachineLearning::Gradient g;
	// n += LEARNING_RATE*this->get_gradient();
}

void MachineLearning::Net::train(const TrainingDataset& data,int iters) {
	for (int i = 0; i < iters; ++i) {
		this->train(data);
	}
}