// #include "net.h"
#include "layer.h"
#include "UnitTest.h"

void la_matrix_tests() {
	LinearAlgebra::Matrix x = {{1},{1},{1},{1}};
	std::cout << x << std::endl;
	std::cout << LinearAlgebra::transpose(x) << std::endl;
}

void net_tests() {
	// LinearAlgebra::Matrix x1 = {{5,5},{5,5}};
	// LinearAlgebra::Matrix x2 = {{1,1},{1,1}};
	// LinearAlgebra::Matrix x3 = {{5,5},{5,5.000001f}};
	// LinearAlgebra::Matrix y1 = {{8,8},{8,8}};
	// LinearAlgebra::Matrix y2 = {{4,4},{4,4}};
	// TEST_RETURN_FUNC(MachineLearning::error(x1,x2),==,y1);
	// TEST_RETURN_FUNC((x1==x3),==,true);
	// TEST_RETURN_FUNC(MachineLearning::error_ddx(x1,x2),==,y2);
}

void activation_function_tests() {
	MachineLearning::ActivationFunction relu = MachineLearning::get_leaky_ReLU();
	TEST_RETURN_FUNC(relu(0.5f),==,0.5f);
	TEST_RETURN_FUNC(relu(-1.0f),==,-0.1f);
	TEST_RETURN_FUNC(relu((LinearAlgebra::Matrix){{-1,2},{3,-4}}),==,((LinearAlgebra::Matrix){{-0.1f,2.0f},{3.0f,-0.4f}}));
}

void LayerParams_tests() {
	LinearAlgebra::Matrix w = {{1,1},{1,1}};
	LinearAlgebra::Matrix b = {{1},{1}};
	LinearAlgebra::VerticalVector x_vv = {1,1};
	LinearAlgebra::Matrix x_m = {{1},{1}};
	MachineLearning::LayerParams lp(w,b);
	LinearAlgebra::transpose(b);
	// MachineLearning::LayerParams lp2(LinearAlgebra::transpose(b),(LinearAlgebra::Matrix){{1}});
	TEST_RETURN_FUNC(lp(x_vv),==,((LinearAlgebra::VerticalVector){3,3}));
	TEST_RETURN_FUNC(lp(x_vv),!=,((LinearAlgebra::VerticalVector){3,3.1}));
	TEST_RETURN_FUNC(lp(x_m),==,((LinearAlgebra::VerticalVector){3,3}));
	TEST_RETURN_FUNC(lp(x_m),!=,((LinearAlgebra::VerticalVector){3,3.1}));
	// TEST_RETURN_FUNC(lp2(b),==,(LinearAlgebra::Matrix){{3}});
}

void layer_tests() {
	// MachineLearning::Layer l(5,5,MachineLearning::get_leaky_ReLU());
	
}

void ForPropIter_tests() {
	
}

int main(int argc, char const *argv[]) {
	la_matrix_tests();
	net_tests();
	activation_function_tests();
	layer_tests();
	LayerParams_tests();
	print_report_card();

	// std::cout << "Let's look for a seg fault\n";
	// LinearAlgebra::Matrix a = {{1,1}};
	// a.at_transpose(MINDEX(0,0));
	return 0;
}