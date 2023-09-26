#include "NetTest.h"

#define H 0.05f

void NetTest::PrivateAPI::calculate_gradient() {
	MachineLearning::NetDef def = {2,1};
	MachineLearning::Net n(def,1.0f); // single-node net with all parameters equal to 1
	LinearAlgebra::Matrix x(MINDEX(2,100)),y(MINDEX(1,100));
	x.set_contents(1.0f);
	y.set_contents(2.0f);
	MachineLearning::TrainingDataset td = {x,y};
	n.load_training_data(td);
	PRINT_VAR(n());
	PRINT_VAR(n.calculate_gradient());

	/**
	 * TODO:	Add a bunch of different tests with neural nets of different sizes.
	 * 			Calculate the expected results by hand
	 */
}

void NetTest::PublicAPI::load_training_data() {

}
void NetTest::PublicAPI::learn() {

}

void NetTest::execute_all_tests() {
	PrivateAPI::calculate_gradient();
	NetTest::PublicAPI::learn();
	NetTest::PublicAPI::load_training_data();
}