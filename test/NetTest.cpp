#include "NetTest.h"

#define H 0.05f

#include <stdint.h>

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
	LinearAlgebra::Matrix x(MINDEX(2,num_data_points)),y(MINDEX(1,num_data_points));
	x.set_contents(1.0f);
	y.set_contents(2.0f);
	MachineLearning::TrainingDataset td = {x,y};
	n.load_training_data(td);
	MachineLearning::Gradient g = make_gradient(n);
	g.front().weights = {{1,1}};
	g.front().biases = {{1}};

	TEST_RETURN_FUNC(n.calculate_gradient(),==,g);
}

void test_calculate_gradient_two_layers(MachineLearning::uint num_data_points) {
	MachineLearning::NetDef def = {4,2,1};
	MachineLearning::Net n(def,1.0f); // single-node net with all parameters equal to 1
	LinearAlgebra::Matrix x(MINDEX(4,num_data_points)),y(MINDEX(1,num_data_points));
	x.set_contents(1.0f);
	y.set_contents(2.0f);
	MachineLearning::TrainingDataset td = {x,y};
	n.load_training_data(td);
	MachineLearning::Gradient g = make_gradient(n);

	TEST_RETURN_FUNC(n.calculate_gradient(),==,g);
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
	NetTest::PrivateAPI::calculate_gradient();
	NetTest::PublicAPI::learn();
	NetTest::PublicAPI::load_training_data();
}