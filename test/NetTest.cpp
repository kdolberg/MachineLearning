#include <fstream>
#include <stdint.h>
#include "NetTest.h"

std::string verify_act_funcs() {
	MachineLearning::NetDef def = {2,5,5,1};
	MachineLearning::Net n(def);
	std::stringstream ss;
	ss << n.afs;
	return ss.str();
}

bool verify_sizes() {
	MachineLearning::NetDef def = {2,5,5,1};
	MachineLearning::Net n(def);

	bool ret = true;

	auto j = def.begin();

	for (auto i = n.begin(); i != n.end(); ++i,++j) {
		ret = ret && (i->get_num_inputs()==(*j));
		ret = ret && (i->get_num_outputs()==(*std::next(j)));
	}
	return ret;
}

void NetTest::PublicAPI::Net_constructors() {
	TEST_RETURN_FUNC(verify_act_funcs(),==,"leaky_ReLU leaky_ReLU sigmoid ");
	TEST_RETURN_FUNC(verify_sizes(),==,true);
}

MachineLearning::Gradient make_gradient(const MachineLearning::Net& n) {
	MachineLearning::Gradient ret;

	for (auto i = n.cbegin(); i != n.cend(); ++i) {
		ret.push_back(*i);
	}

	return ret;
}

void test_calculate_gradient_one_node_net_uniform_data(MachineLearning::uint num_data_points) {
	MachineLearning::NetDef def = {2,1};
	MachineLearning::Net n(def,1.0f); // single-node net with all parameters equal to 1
	n.learning_rate = 1;
	n.afs = (std::list<MachineLearning::ActivationFunction>){MachineLearning::get_leaky_ReLU()};
	std::cout << n.afs << std::endl;
	LinearAlgebra::Matrix x(MINDEX(2,num_data_points)),y(MINDEX(1,num_data_points));
	x.set_contents(1.0f);
	y.set_contents(2.0f);
	MachineLearning::TrainingDataset td = {x,y};
	n.load_training_data(td);
	MachineLearning::Gradient g = make_gradient(n);
	g.front().weights = {{1,1}};
	g.front().biases = {{1}};

	TEST_VOID_FUNC(n.calculate_gradient(),n.get_partial_derivatives(),==,g);
}

void test_calculate_gradient_two_layers(MachineLearning::uint num_data_points) {
	MachineLearning::uint num_inputs = 1;
	MachineLearning::uint num_outputs = 1;
	MachineLearning::NetDef def = {num_inputs,num_inputs,num_outputs};
	MachineLearning::Net n(def,1.0f); // two-layer net
	n.afs = (std::list<MachineLearning::ActivationFunction>){MachineLearning::get_leaky_ReLU(), MachineLearning::get_leaky_ReLU()};
	LinearAlgebra::Matrix x(MINDEX(num_inputs,num_data_points)),y(MINDEX(1,num_data_points));
	x.set_contents(1.0f);
	y.set_contents(2.0f);
	MachineLearning::TrainingDataset td = {x,y};
	n.load_training_data(td);
	MachineLearning::LayerParams lp_input,lp_output;
	lp_output = {(LinearAlgebra::Matrix){{2.0f}},(LinearAlgebra::Matrix){1.0f}};
	lp_input = {(LinearAlgebra::Matrix){{1.0f}},(LinearAlgebra::Matrix){1.0f}};
	MachineLearning::Gradient g;
	// DOUBLE-CHECK THESE "CORRECT" OUTPUTS.
	g.push_back(lp_input);
	g.push_back(lp_output);

	TEST_VOID_FUNC(n.calculate_gradient(),n.get_partial_derivatives(),==,g);
}

void test_calculate_gradient(MachineLearning::uint num_data_points) {
	test_calculate_gradient_one_node_net_uniform_data(num_data_points);
	test_calculate_gradient_two_layers(num_data_points);
}

void NetTest::PrivateAPI::calculate_gradient() {
	test_calculate_gradient(1);
	test_calculate_gradient(10);
	test_calculate_gradient(100);
	test_calculate_gradient(7);
}

void NetTest::PublicAPI::load_training_data() {

}
void NetTest::PublicAPI::learn() {

}

void NetTest::execute_all_tests() {
	NetTest::PublicAPI::Net_constructors();
	NetTest::PrivateAPI::calculate_gradient();
	// NetTest::PublicAPI::learn();
	NetTest::PublicAPI::load_training_data();
}