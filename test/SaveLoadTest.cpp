#include "machine_learning.h"
#include "UnitTest.h"
#include "SaveLoadTest.h"

int static get_file_num() {
	static int i = 0;
	return i++;
}

template <typename T>
void static test_list(std::list<T> l1) {
	std::string str = std::string("test/saves/list") + std::to_string(get_file_num()) + std::string(".list");
	std::list<T> l2;
	TEST_RETURN_FUNC(MachineLearning::save(l1,str.c_str()),==,true);
	TEST_RETURN_FUNC(MachineLearning::load(l2,str.c_str()),==,true);
	TEST_RETURN_FUNC(l1,==,l2);
}

void SaveLoadTest::LayerParams() {
	PRINT_FUNC();
	LinearAlgebra::Matrix w(5,true),b(MINDEX(5,1));
	b[MINDEX(0,0)] = 1;
	MachineLearning::LayerParams lp1 = {w,b};
	MachineLearning::save(lp1,"test/saves/1.lp");
	MachineLearning::LayerParams lp2;
	MachineLearning::load(lp2,"test/saves/1.lp");
	TEST_RETURN_FUNC(lp1,==,lp2);
}

void SaveLoadTest::Matrix() {
	PRINT_FUNC();
	LinearAlgebra::Matrix m1(5,true), m2;
	MachineLearning::save(m1,"test/saves/m.mat");
	MachineLearning::load(m2,"test/saves/m.mat");
	TEST_RETURN_FUNC(m1,==,m2);
}

void SaveLoadTest::TrainingDataset() {
	PRINT_FUNC();
	{
		LinearAlgebra::Matrix x(5,true),y(MINDEX(5,1));
		x.randomize();
		y.randomize();
		MachineLearning::TrainingDataset td1 = {x,y},td2;
		TEST_RETURN_FUNC(MachineLearning::save(td1,"test/saves/td0.td"),==,true);
		TEST_RETURN_FUNC(MachineLearning::load(td2,"test/saves/td0.td"),==,true);
		TEST_RETURN_FUNC(td1,==,td2);
	} {
		MachineLearning::TrainingDataset td1,td2;
		TEST_RETURN_FUNC(MachineLearning::save(td1,"test/saves/td1.td"),==,true);
		TEST_RETURN_FUNC(MachineLearning::load(td2,"test/saves/td1.td"),==,true);
		TEST_RETURN_FUNC(td1,==,td2);
	}
}

void SaveLoadTest::Net() {
	PRINT_FUNC();
	{
		MachineLearning::Net n1(MachineLearning::NetDef({5,4,3,2,1})), n2;
		LinearAlgebra::Matrix x(5,true),y(MINDEX(5,1));
		x.randomize();
		y.randomize();
		MachineLearning::TrainingDataset td1 = {x,y};
		n1.load_training_data(td1);
		TEST_RETURN_FUNC(MachineLearning::save(n1,"test/saves/testnet.nn"),==,true);
		TEST_RETURN_FUNC(MachineLearning::load(n2,"test/saves/testnet.nn"),==,true);
		TEST_RETURN_FUNC(n1.get_activation_function_list(),==,n2.get_activation_function_list());
		TEST_RETURN_FUNC(n1.get_training_data(),==,n2.get_training_data());
		TEST_RETURN_FUNC(n1,==,n2);
	} {
		MachineLearning::Net n1(MachineLearning::NetDef({5,4,3,2,1})), n2;
		LinearAlgebra::Matrix x(5,true),y(MINDEX(5,1));
		x.randomize();
		y.randomize();
		TEST_RETURN_FUNC(MachineLearning::save(n1,"test/saves/testnet.nn"),==,true);
		TEST_RETURN_FUNC(MachineLearning::load(n2,"test/saves/testnet.nn"),==,true);
		TEST_RETURN_FUNC(n1.get_activation_function_list(),==,n2.get_activation_function_list());
		TEST_RETURN_FUNC(n1.get_training_data(),==,n2.get_training_data());
		TEST_RETURN_FUNC(n1,==,n2);
	}
}

void SaveLoadTest::ActivationFunction() {
	PRINT_FUNC();
	{
		MachineLearning::ActivationFunction af1, af2;
		af1 = MachineLearning::get_sigmoid();
		TEST_RETURN_FUNC(MachineLearning::save(af1,"test/saves/sigmoid.af"),==,true);
		TEST_RETURN_FUNC(MachineLearning::load(af2,"test/saves/sigmoid.af"),==,true);
		TEST_RETURN_FUNC(af1.str(),==,SIGMOID_NAME);
		TEST_RETURN_FUNC(af2.str(),==,SIGMOID_NAME);
		TEST_RETURN_FUNC(af1,==,af2);
	}{
		MachineLearning::ActivationFunction af1, af2;
		af1 = MachineLearning::get_leaky_ReLU();
		TEST_RETURN_FUNC(MachineLearning::save(af1,"test/saves/leakyrelu.af"),==,true);
		TEST_RETURN_FUNC(MachineLearning::load(af2,"test/saves/leakyrelu.af"),==,true);
		TEST_RETURN_FUNC(af1.str(),==,LEAKY_RELU_NAME);
		TEST_RETURN_FUNC(af2.str(),==,LEAKY_RELU_NAME);
		TEST_RETURN_FUNC(af1,==,af2);
	}
}

#define TEST_LIST(__type__,__var__) std::cout << #__type__ << std::endl;\
									test_list<__type__>((std::list<__type__>) __var__);

void SaveLoadTest::list() {
	PRINT_FUNC();
	MachineLearning::Net n(MachineLearning::NetDef({5,4,3,2,1}));
	TEST_LIST(MachineLearning::LayerParams,n);
	TEST_LIST(MachineLearning::ActivationFunction,n.get_activation_function_list());
	// test_list<MachineLearning::LayerParams>((std::list<MachineLearning::LayerParams>)n);
	// test_list<MachineLearning::ActivationFunction>(n.get_activation_function_list());
}

void SaveLoadTest::execute_all_tests() {
	SaveLoadTest::list();
	SaveLoadTest::LayerParams();
	SaveLoadTest::Matrix();
	SaveLoadTest::TrainingDataset();
	SaveLoadTest::ActivationFunction();
	SaveLoadTest::Net();
}