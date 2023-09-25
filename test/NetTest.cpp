#include "NetTest.h"

#define H 0.00001f

MachineLearning::scalar numerical_derivative(MachineLearning::ActivationFunction af, MachineLearning::scalar x) {
	MachineLearning::scalar y1 = af(x);
	MachineLearning::scalar y2 = af(x+H);
	return (y2-y1)/H;
}

MachineLearning::scalar NetTest::PrivateAPI::numerical_derivative(MachineLearning::Net& n,MachineLearning::scalar * wb_ptr) {
	n.forward_propagate();
	MachineLearning::scalar E1 = n.error();
	MachineLearning::scalar tmp = (*wb_ptr);
	(*wb_ptr) += H;
	n.forward_propagate();
	MachineLearning::scalar E2 = n.error();
	(*wb_ptr) = tmp;
	return ((E2-E1)/H);
}

void NetTest::numerical_derivative_test() {
	MachineLearning::ActivationFunction sig = MachineLearning::get_sigmoid();
	MachineLearning::scalar x = 0.5f;
	TEST_RETURN_FUNC(std::abs(sig.ddx(x)-numerical_derivative(sig,x)),<,0.05f);
}

MachineLearning::Gradient NetTest::PrivateAPI::numerical_gradient(MachineLearning::Net& n) {
	MachineLearning::Gradient ret;
	for (MachineLearning::Net::iterator i = n.begin(); i != n.end(); ++i) {
		LinearAlgebra::Matrix w_tmp(i->weights.size()), b_tmp(i->biases.size());
		for (MachineLearning::mindex wbi = {0,0}; wbi.row < i->weights.get_num_rows(); ++wbi.row) {
			for (wbi.col = 0; wbi.col < i->weights.get_num_cols(); ++wbi.col) {
				w_tmp[wbi] = numerical_derivative(n,&(i->weights[wbi]));
			}
			MachineLearning::mindex bbi = MINDEX(wbi.row,0);
			b_tmp[bbi] = numerical_derivative(n,&(i->biases[bbi]));
		}
		ret.push_back(MachineLearning::LayerParams({w_tmp,b_tmp}));
	}
	return ret;
}

LinearAlgebra::Matrix percent_error(const LinearAlgebra::Matrix& a, const LinearAlgebra::Matrix& b) {
	assert(a.size()==b.size());
	LinearAlgebra::Matrix ret(a.size());
	for (MachineLearning::mindex i = {0,0}; i.row < a.get_num_rows(); ++i.row) {
		for (i.col = 0; i.col < a.get_num_cols(); ++i.col) {
			ret[i] = 100.0f*(a[i]-b[i])/a[i];
		}
	}
	return ret;
}

MachineLearning::LayerParams percent_error(const MachineLearning::LayerParams& a, const MachineLearning::LayerParams& b) {
	LinearAlgebra::Matrix w_ret = percent_error(a.get_weights(),b.get_weights());
	LinearAlgebra::Matrix b_ret = percent_error(a.get_biases(),b.get_biases());
	MachineLearning::LayerParams ret = {w_ret,b_ret};
	return ret;
}

MachineLearning::Gradient percent_error(const MachineLearning::Gradient& a, const MachineLearning::Gradient& b) {
	assert(a.size()==b.size());
	MachineLearning::Gradient ret;

	for (auto a_iter = a.begin(), b_iter = b.begin(); a_iter != a.end();) {
		ret.push_back(percent_error(*a_iter,*b_iter));
		++a_iter;
		++b_iter;
	}
	return ret;
}

void NetTest::PrivateAPI::calculate_gradient() {
	MachineLearning::NetDef def = {2,5,8,2,2,2,1};
	MachineLearning::Net n(def,1.0f);
	LinearAlgebra::Matrix x={{1.0f,1.0f},{1.0f,1.0f}}, y={{1.0f,1.0f}};
	MachineLearning::TrainingDataset td = {x,y};
	n.load_training_data(td);

	PRINT_VAR(percent_error(n.calculate_gradient(),numerical_gradient(n)));

	TEST_RETURN_FUNC(n.calculate_gradient(),==,numerical_gradient(n));
}

void calculate_gradient_random() {
	MachineLearning::NetDef def = {2,5,8,2,2,2,1};
	MachineLearning::Net n(def,true);
	LinearAlgebra::uint num_data_points = 10;
	LinearAlgebra::Matrix x(MINDEX(n.get_num_inputs(),num_data_points)), y(MINDEX(n.get_num_outputs(),num_data_points));
	x.randomize();
	y.randomize();
	PRINT_VAR(y);
	MachineLearning::TrainingDataset td = {x,y};
	n.load_training_data(td);

	// TEST_RETURN_FUNC(n.calculate_gradient(),==,numerical_gradient(n));

	PRINT_VAR(percent_error(n.calculate_gradient(),NetTest::PrivateAPI::numerical_gradient(n)));
	PRINT_VAR(n.calculate_gradient());
	PRINT_VAR(NetTest::PrivateAPI::numerical_gradient(n));
}

void NetTest::PublicAPI::load_training_data() {

}
void NetTest::PublicAPI::learn() {

}

void NetTest::execute_all_tests() {
	NetTest::numerical_derivative_test();
	calculate_gradient_random();
	NetTest::PublicAPI::learn();
	NetTest::PublicAPI::load_training_data();
}