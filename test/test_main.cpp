#include "net.h"
#include "UnitTest.h"

void net_tests() {
	LinearAlgebra::Matrix x1 = {{5,5},{5,5}};
	LinearAlgebra::Matrix x2 = {{1,1},{1,1}};
	LinearAlgebra::Matrix x3 = {{5,5},{5,5.000001f}};
	LinearAlgebra::Matrix y1 = {{8,8},{8,8}};
	LinearAlgebra::Matrix y2 = {{4,4},{4,4}};
	TEST_RETURN_FUNC(MachineLearning::error(x1,x2),==,y1);
	TEST_RETURN_FUNC((x1==x3),==,true);
	TEST_RETURN_FUNC(MachineLearning::error_ddx(x1,x2),==,y2);
}

void activation_function_tests() {
	MachineLearning::ActivationFunction relu = MachineLearning::get_leaky_ReLU();
	TEST_RETURN_FUNC(relu(0.5f),==,0.5f);
	TEST_RETURN_FUNC(relu(-1.0f),==,-0.1f);
	TEST_RETURN_FUNC(relu((LinearAlgebra::Matrix){{-1,2},{3,-4}}),==,((LinearAlgebra::Matrix){{-0.1f,2.0f},{3.0f,-0.4f}}));
}

void layer_tests() {
	MachineLearning::Layer l(5,5,MachineLearning::get_leaky_ReLU());
	std::cout << l.parameters << std::endl;
}

int main(int argc, char const *argv[]) {
	net_tests();
	activation_function_tests();
	layer_tests();
	print_report_card();

	return 0;
}