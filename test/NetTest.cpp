#include "NetTest.h"

#define H 0.001f



LinearAlgebra::scalar_t numerical_derivative(MachineLearning::ActivationFunction af, LinearAlgebra::scalar_t x) {
	LinearAlgebra::scalar_t y1 = af(x);
	LinearAlgebra::scalar_t y2 = af(x+H);
	return (y2-y1)/H;
}

LinearAlgebra::scalar_t NetTest::PrivateAPI::numerical_derivative(MachineLearning::Net& n,MachineLearning::scalar_t * wb_ptr) {
	n.forward_propagate();
	LinearAlgebra::scalar_t E1 = n.error();
	LinearAlgebra::scalar_t tmp = (*wb_ptr);
	(*wb_ptr) += H;
	n.forward_propagate();
	LinearAlgebra::scalar_t E2 = n.error();
	(*wb_ptr) = tmp;
	return ((E2-E1)/H);
}

void NetTest::numerical_derivative_test() {
	MachineLearning::ActivationFunction sig = MachineLearning::get_sigmoid();
	LinearAlgebra::scalar_t x = 0.5f;
	TEST_RETURN_FUNC(std::abs(sig.ddx(x)-numerical_derivative(sig,x)),<,0.05f);
}

MachineLearning::Gradient NetTest::PrivateAPI::numerical_gradient(MachineLearning::Net& n) {
	MachineLearning::Gradient ret;
	for (MachineLearning::Net::iterator i = n.begin(); i != n.end(); ++i) {
		LinearAlgebra::Matrix w_tmp(i->weights.size()), b_tmp(i->biases.size());
		for (LinearAlgebra::mindex_t wbi = {0,0}; wbi.row < i->weights.get_num_rows(); ++wbi.row) {
			for (wbi.col = 0; wbi.col < i->weights.get_num_cols(); ++wbi.col) {
				w_tmp[wbi] = numerical_derivative(n,&(i->weights[wbi]));
			}
			LinearAlgebra::mindex_t bbi = MINDEX(wbi.row,0);
			b_tmp[bbi] = numerical_derivative(n,&(i->biases[bbi]));
		}
		ret.push_back(MachineLearning::LayerParams({w_tmp,b_tmp}));
	}
	return ret;
}

void NetTest::PrivateAPI::calculate_gradient() {
	MachineLearning::NetDef def = {2,2,1};
	MachineLearning::Net n(def,0.5f);
	LinearAlgebra::Matrix x={{0.5f,1.0f},{-0.5f,0.6f}}, y={{5.0f,0.1f}};
	MachineLearning::TrainingDataset td = {x,y};
	n.load_training_data(td);
	TEST_RETURN_FUNC(n.calculate_gradient(),==,numerical_gradient(n));
}

void NetTest::PublicAPI::load_training_data() {

}
void NetTest::PublicAPI::learn() {

}

void NetTest::execute_all_tests() {
	NetTest::numerical_derivative_test();
	NetTest::PrivateAPI::calculate_gradient();
	NetTest::PublicAPI::learn();
	NetTest::PublicAPI::load_training_data();
}