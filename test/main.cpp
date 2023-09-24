#include "main.h"

void la_matrix_tests() {
	LinearAlgebra::Matrix m;
	LinearAlgebra::Matrix n(MINDEX(5,4));
	TEST_RETURN_FUNC(m.empty(),==,true);
	TEST_RETURN_FUNC(n.empty(),==,false);
}

void activation_function_tests() {
	MachineLearning::ActivationFunction relu = MachineLearning::get_leaky_ReLU();
	TEST_RETURN_FUNC(relu(0.5f),==,0.5f);
	TEST_RETURN_FUNC(relu(-1.0f),==,-0.1f);
	TEST_RETURN_FUNC(relu((LinearAlgebra::Matrix){{-1,2},{3,-4}}),==,((LinearAlgebra::Matrix){{-0.1f,2.0f},{3.0f,-0.4f}}));
}

bool LayerParams_randomize_randomizes_correctly() {
	bool ret = true;
	MachineLearning::LayerParams lp1({{1,1},{1,1}},{{1},{1}});
	lp1.randomize();
	MachineLearning::LayerParams lp2 = lp1;
	for (int i = 0; i < 1000; ++i) {
		lp1.randomize();
		lp2.randomize();
		ret = ret && (lp1!=lp2);
		ret = ret && (lp1.get_weights().size()==lp2.get_weights().size());
		ret = ret && (lp1.get_biases().size()==lp2.get_biases().size());
	}
	return true;
}

void LayerParams_tests() {
	{
		LinearAlgebra::Matrix w = {{1,1},{1,1}};
		LinearAlgebra::Matrix b = {{1},{1}};
		LinearAlgebra::VerticalVector x_vv = {1,1};
		LinearAlgebra::Matrix x_m = {{1},{1}};
		MachineLearning::LayerParams lp(w,b);
		TEST_RETURN_FUNC(lp(x_vv),==,((LinearAlgebra::VerticalVector){3,3}));
		TEST_RETURN_FUNC(lp(x_vv),!=,((LinearAlgebra::VerticalVector){3,3.1}));
		TEST_RETURN_FUNC(lp(x_m),==,((LinearAlgebra::VerticalVector){3,3}));
		TEST_RETURN_FUNC(lp(x_m),!=,((LinearAlgebra::VerticalVector){3,3.1}));
	}
	TEST_RETURN_FUNC(LayerParams_randomize_randomizes_correctly(),==,true);
	{
		MachineLearning::LayerParams lp({{1,1},{1,1},{1,1}},{{1},{1},{1}});
		TEST_RETURN_FUNC(lp.get_num_inputs(),==,lp.get_weights().get_num_cols());
		TEST_RETURN_FUNC(lp.get_num_outputs(),==,lp.get_weights().get_num_rows());
		TEST_RETURN_FUNC(lp.get_num_outputs(),==,lp.get_biases().get_num_rows());
		TEST_RETURN_FUNC(lp.get_num_outputs(),==,3);
		TEST_RETURN_FUNC(lp.get_num_inputs(),==,2);
	} {
		MachineLearning::LayerParams lp1(
		{// w
			{0.4,0.2,10},
			{5,2,8},
			{10,3,1}
		},{// b
			{82.5},
			{7.9},
			{-50.4}
		});
		MachineLearning::LayerParams lp2(
		{// w
			{0.4,0.2,10},
			{5,2,8},
			{10,3,1}
		},{// b
			{82.5},
			{7.9},
			{-50.4}
		});
		MachineLearning::LayerParams lp3(
		{// w
			{0.4,0.2,10 + TOLERANCE*0.9},
			{5,2,8},
			{10,3,1 + TOLERANCE*0.9}
		},{// b
			{82.5},
			{7.9 - TOLERANCE*0.9},
			{-50.4}
		});
		MachineLearning::LayerParams lp4(
		{// w
			{0.4,0.2,10 + TOLERANCE*1.1},
			{5,2,8},
			{10,3,1}
		},{// b
			{82.5},
			{7.9},
			{-50.4}
		});
		TEST_RETURN_FUNC(lp1,==,lp2);
		TEST_RETURN_FUNC(lp1,==,lp3);
		TEST_RETURN_FUNC(lp1,!=,lp4);
	}
}

MachineLearning::Net make_uniform_net(LinearAlgebra::scalar_t num) {
	return MachineLearning::Net();
}

void ForProp_test() {
	{
		// MachineLearning::NetDef def = {2,1};
		// MachineLearning::Net n(def,1.0f);
		// MachineLearning::ForPropIter fpi = (MachineLearning::ForPropIter)n.std::list<MachineLearning::LayerStruct>::begin();
		// LinearAlgebra::Matrix input = {{1},{1}};
		// TEST_VOID_FUNC(fpi.update_data_cache(input),fpi.get_post_act_func_output(),==,((LinearAlgebra::Matrix){{3}}));
		// TEST_RETURN_FUNC(n.forward_propagate(input),==,((LinearAlgebra::Matrix){{3}}));
	}
}

void PropIter_tests() {
	// ForProp_test();
	// LinearAlgebra::Matrix x_in = {{1,1},{5,5},{10,10},{0.5,0.5}};
	// LinearAlgebra::Matrix y_out = {{5},{5},{5},{5}};
	// LinearAlgebra::Matrix y_incorrect = {{1},{1},{1},{1}};

	// MachineLearning::TrainingDataset td = {LinearAlgebra::transpose(x_in),LinearAlgebra::transpose(y_out)};
	// std::vector<MachineLearning::uint> def = {2,1,1,1,1,1};
	// MachineLearning::Net n(def,true);
	// MachineLearning::ForPropIter fpi = (MachineLearning::ForPropIter)n.std::list<MachineLearning::LayerStruct>::begin();
	// fpi.update_data_cache(td.x);
	// ++fpi;
	// for(; fpi != (MachineLearning::ForPropIter)n.std::list<MachineLearning::LayerStruct>::end(); ++fpi) {
	// 	fpi.update_data_cache();
	// }
	// MachineLearning::BackPropIter bpi = (MachineLearning::BackPropIter)n.std::list<MachineLearning::LayerStruct>::rbegin();
	// bpi.update_data_cache_output_layer(LinearAlgebra::transpose(y_incorrect) - td.y);
	// ++bpi;
	// for(; bpi != std::prev(n.std::list<MachineLearning::LayerStruct>::rend()); ++bpi) {
	// 	bpi.update_data_cache();
	// }
	// bpi.update_data_cache_input_layer(td.x);
}

void tmp() {
	LinearAlgebra::Matrix m = {{1,2},{3,4}};
	PRINT_VAR(m);
	LinearAlgebra::mindex_t i = {0,1};
	PRINT_VAR(m[i]);
	LinearAlgebra::uint j = 0;
	LinearAlgebra::uint k = 0;
	LinearAlgebra::ref_mindex l = {j,k};

	for (j = 0; j < m.get_num_rows(); ++j) {
		for (k = 0; k < m.get_num_cols(); ++k) {
			std::cout << m[l] << " ";
		}
		std::cout << std::endl;
	}
}

int main(int argc, char const *argv[]) {
	// la_matrix_tests();
	NetTest::execute_all_tests();
	print_report_card();
	return 0;
}