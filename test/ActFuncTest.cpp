#include "ActFuncTest.h"

void ActFuncTest::str() {
	MachineLearning::ActivationFunction relu = MachineLearning::get_leaky_ReLU();
	MachineLearning::ActivationFunction sig = MachineLearning::get_sigmoid();
	TEST_RETURN_FUNC(relu.str(),==,"leaky_ReLU");
	TEST_RETURN_FUNC(sig.str(),==,"sigmoid");
}

void ActFuncTest::leaky_ReLU() {
	MachineLearning::ActivationFunction relu = MachineLearning::get_leaky_ReLU();
	TEST_RETURN_FUNC(relu(0.5f),==,0.5f);
	TEST_RETURN_FUNC(relu(-1.0f),==,-0.1f);
	TEST_RETURN_FUNC(relu((LinearAlgebra::Matrix){{-1,2},{3,-4}}),==,((LinearAlgebra::Matrix){{-0.1f,2.0f},{3.0f,-0.4f}}));
	TEST_RETURN_FUNC(relu.ddx(5.0f),==,1.0f);
	TEST_RETURN_FUNC(relu.ddx(-7.0f),==,0.1f);
}

void ActFuncTest::sigmoid() {
	MachineLearning::ActivationFunction sigm = MachineLearning::get_sigmoid();
	MachineLearning::scalar ans;
	ans = 0.0f;
	TEST_VOID_FUNC(ans=sigm(0.5f),		LinearAlgebra::approx(ans,0.6224593312018958f),		==, true);
	TEST_VOID_FUNC(ans=sigm(-1.0f),		LinearAlgebra::approx(ans,0.26894142136992605f),	==, true);
	TEST_VOID_FUNC(ans=sigm.ddx(5.0f),	LinearAlgebra::approx(ans,0.006648056670778641f),	==, true);
	TEST_VOID_FUNC(ans=sigm.ddx(-7.0f),	LinearAlgebra::approx(ans,0.000910221180119593f),	==, true);
}

void ActFuncTest::execute_all_tests() {
	ActFuncTest::str();
	ActFuncTest::leaky_ReLU();
	ActFuncTest::sigmoid();
}