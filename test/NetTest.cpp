#include "NetTest.h"

#define H 0.001f



LinearAlgebra::scalar_t numerical_derivative(MachineLearning::ActivationFunction af, LinearAlgebra::scalar_t x) {\
	LinearAlgebra::scalar_t y1 = af(x);
	LinearAlgebra::scalar_t y2 = af(x+H);
	return (y2-y1)/H;
}

void NetTest::numerical_derivative_test() {
	MachineLearning::ActivationFunction sig = MachineLearning::get_sigmoid();
	LinearAlgebra::scalar_t x = 0.5f;
	TEST_RETURN_FUNC(std::abs(sig.ddx(x)-numerical_derivative(sig,x)),<,0.05f);
}

MachineLearning::Gradient NetTest::PrivateAPI::numerical_gradient_ddx(MachineLearning::Net& n,const MachineLearning::TrainingDataset& td) {
	MachineLearning::Gradient ret;
	for (MachineLearning::Net::iterator i = n.begin(); i != n.end(); ++i) {
		LinearAlgebra::Matrix w_tmp(i->params.weights.size()), b_tmp(i->params.biases.size());
		for (LinearAlgebra::mindex_t wbi = {0,0}; wbi.row < i->params.weights.get_num_rows(); ++wbi.row) {
			for (wbi.col = 0; wbi.col < i->params.weights.get_num_cols(); ++wbi.col) {
				// w_tmp = numerical_gradient_ddx(n,td,&(i->params.weights[wbi]));
			}
			// b_tmp = numerical_gradient_ddx(n,td,&(i->params.biases[MINDEX(wbi.row,0)]));
		}
		ret.push_back(MachineLearning::LayerParams({w_tmp,b_tmp}));
	}
	return ret;
}

void NetTest::PrivateAPI::calculate_gradient() {
	MachineLearning::NetDef def = {2,2,1};
	MachineLearning::Net n(def,0.5f);
	LinearAlgebra::Matrix x={{0.5f,1.0f},{-0.5f,0.6f}}, y={{5.0f},{0.1f}};
	MachineLearning::TrainingDataset td = {x,y};
	n.load_training_data(td);
	n.calculate_gradient();
	// TEST_RETURN_FUNC(n.calculate_gradient(),==,numerical_gradient_ddx(n,td));
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