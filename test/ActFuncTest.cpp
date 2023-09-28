#include "ActFuncTest.h"

void ActFuncTest::PublicAPI::str() {
	MachineLearning::ActivationFunction relu = MachineLearning::get_leaky_ReLU();
	MachineLearning::ActivationFunction sig = MachineLearning::get_sigmoid();
	TEST_RETURN_FUNC(relu.str(),==,"leaky_ReLU");
	TEST_RETURN_FUNC(sig.str(),==,"sigmoid");
}

void ActFuncTest::execute_all_tests() {
	ActFuncTest::PublicAPI::str();
}