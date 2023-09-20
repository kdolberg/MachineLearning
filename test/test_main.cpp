// #include "net.h"
#include "layer.h"
#include "UnitTest.h"
#include "net.h"

void la_matrix_tests() {
	LinearAlgebra::Matrix x = {{1},{1},{1},{1}};
	std::cout << x << std::endl;
	std::cout << LinearAlgebra::transpose(x) << std::endl;
}

// typedef struct NetUintIter {
// 	std::vector<MachineLearning::uint>::const_iterator def;
// 	MachineLearning::Net::const_iterator n;
// 	NetUintIter& operator++() {
// 		++this->def;
// 		++this->n;
// 		return (*this);
// 	}
// 	bool operator==(const NetUintIter& iter) {
// 		if((iter.n)==(this->n)) {
// 			assert((std::next(iter.def)==this->def) || (iter.def==std::next(this->def)));
// 			return true;
// 		} else {
// 			return false;
// 		}
// 	}
// } NetUintIter;

// bool Net_constructor_gives_correct_num_inputs() {
// 	std::vector<MachineLearning::uint> def = {5,4,3,2,1};
// 	MachineLearning::Net n(def);
// 	bool all_nums_correct = true;
// 	for (NetUintIter i = {def.cbegin(),n.cbegin()}; i != (NetUintIter){def.cend(),n.cend()}; ++i) {
// 		std::cout << "# inputs: " << i.n->params.get_num_inputs() << std::endl;
// 		std::cout << (i.n)->params << std::endl;
// 		all_nums_correct = all_nums_correct && (i.n->params.get_num_inputs()==*(i.def));
// 	}
// 	return all_nums_correct;
// }

// void Net_tests() {
// 	TEST_RETURN_FUNC(Net_constructor_gives_correct_num_inputs(),==,true);

// }

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

void PropIter_tests() {
	LinearAlgebra::Matrix x_in_sidways = {{1,1},{5,5},{10,10},{0.5,0.5}};
	LinearAlgebra::Matrix y_out = {{5},{5},{5},{5}};
	LinearAlgebra::Matrix y_correct = {{1},{1},{1},{1}};

	MachineLearning::TrainingDataset td = {LinearAlgebra::transpose(x_in_sidways),y_out};
	std::vector<MachineLearning::uint> def = {2,1,1,1,1,1};
	MachineLearning::Net n(def,true);

	std::cout << "FORWARD\n";
	MachineLearning::ForPropIter fpi = n.begin();
	fpi.update_data_cache(td.x);
	++fpi;
	for(; fpi != n.end(); ++fpi) {
		std::cout << fpi->params << std::endl;
		fpi.update_data_cache();
	}
	MachineLearning::BackPropIter bpi = n.rbegin();
	bpi.update_data_cache_output_layer(y_correct - td.y);
	++bpi;
	std::cout << "BACKWARD\n";
	for(; bpi != std::prev(n.rend()); ++bpi) {
		std::cout << bpi->params << std::endl;
		try {
			bpi.update_data_cache();
		} catch (ConfirmationFailure& e) {
			std::cerr << e.what() << std::endl;
		}
	}
	bpi.update_data_cache_input_layer(td.x);
}

int main(int argc, char const *argv[]) {
	la_matrix_tests();
	// Net_tests();
	activation_function_tests();
	LayerParams_tests();
	print_report_card();
	PropIter_tests();
	return 0;
}