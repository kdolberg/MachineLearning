#include "main.h"

// void la_matrix_tests() {
// 	LinearAlgebra::Matrix m;
// 	LinearAlgebra::Matrix n(MINDEX(5,4));
// 	TEST_RETURN_FUNC(m.empty(),==,true);
// 	TEST_RETURN_FUNC(n.empty(),==,false);
// }

// bool LayerParams_randomize_randomizes_correctly() {
// 	bool ret = true;
// 	MachineLearning::LayerParams lp1({{1,1},{1,1}},{{1},{1}});
// 	lp1.randomize();
// 	MachineLearning::LayerParams lp2 = lp1;
// 	for (int i = 0; i < 1000; ++i) {
// 		lp1.randomize();
// 		lp2.randomize();
// 		ret = ret && (lp1!=lp2);
// 		ret = ret && (lp1.get_weights().size()==lp2.get_weights().size());
// 		ret = ret && (lp1.get_biases().size()==lp2.get_biases().size());
// 	}
// 	return true;
// }

// void LayerParams_tests() {
// 	{
// 		LinearAlgebra::Matrix w = {{1,1},{1,1}};
// 		LinearAlgebra::Matrix b = {{1},{1}};
// 		LinearAlgebra::VerticalVector x_vv = {1,1};
// 		LinearAlgebra::Matrix x_m = {{1},{1}};
// 		MachineLearning::LayerParams lp(w,b);
// 		TEST_RETURN_FUNC(lp(x_vv),==,((LinearAlgebra::VerticalVector){3,3}));
// 		TEST_RETURN_FUNC(lp(x_vv),!=,((LinearAlgebra::VerticalVector){3,3.1}));
// 		TEST_RETURN_FUNC(lp(x_m),==,((LinearAlgebra::VerticalVector){3,3}));
// 		TEST_RETURN_FUNC(lp(x_m),!=,((LinearAlgebra::VerticalVector){3,3.1}));
// 	}
// 	TEST_RETURN_FUNC(LayerParams_randomize_randomizes_correctly(),==,true);
// 	{
// 		MachineLearning::LayerParams lp({{1,1},{1,1},{1,1}},{{1},{1},{1}});
// 		TEST_RETURN_FUNC(lp.get_num_inputs(),==,lp.get_weights().get_num_cols());
// 		TEST_RETURN_FUNC(lp.get_num_outputs(),==,lp.get_weights().get_num_rows());
// 		TEST_RETURN_FUNC(lp.get_num_outputs(),==,lp.get_biases().get_num_rows());
// 		TEST_RETURN_FUNC(lp.get_num_outputs(),==,3);
// 		TEST_RETURN_FUNC(lp.get_num_inputs(),==,2);
// 	} {
// 		MachineLearning::LayerParams lp1(
// 		{// w
// 			{0.4,0.2,10},
// 			{5,2,8},
// 			{10,3,1}
// 		},{// b
// 			{82.5},
// 			{7.9},
// 			{-50.4}
// 		});
// 		MachineLearning::LayerParams lp2(
// 		{// w
// 			{0.4,0.2,10},
// 			{5,2,8},
// 			{10,3,1}
// 		},{// b
// 			{82.5},
// 			{7.9},
// 			{-50.4}
// 		});
// 		MachineLearning::LayerParams lp3(
// 		{// w
// 			{0.4,0.2,10 + TOLERANCE*0.9},
// 			{5,2,8},
// 			{10,3,1 + TOLERANCE*0.9}
// 		},{// b
// 			{82.5},
// 			{7.9 - TOLERANCE*0.9},
// 			{-50.4}
// 		});
// 		MachineLearning::LayerParams lp4(
// 		{// w
// 			{0.4,0.2,10 + TOLERANCE*1.1},
// 			{5,2,8},
// 			{10,3,1}
// 		},{// b
// 			{82.5},
// 			{7.9},
// 			{-50.4}
// 		});
// 		TEST_RETURN_FUNC(lp1,==,lp2);
// 		TEST_RETURN_FUNC(lp1,==,lp3);
// 		TEST_RETURN_FUNC(lp1,!=,lp4);
// 	}
// }

// MachineLearning::Net make_uniform_net(MachineLearning::scalar num) {
// 	return MachineLearning::Net();
// }

int main(int argc, char const *argv[]) {
	ActFuncTest::execute_all_tests();
	NetTest::execute_all_tests();
	// NetTest::Performance::execute_all_tests();
	print_report_card();
	return 0;
}