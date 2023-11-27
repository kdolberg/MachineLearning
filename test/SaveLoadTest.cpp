#include "machine_learning.h"
#include "SaveLoadTest.h"

static int get_file_num() {
	static int i = 0;
	return i++;
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
	LinearAlgebra::Matrix x(5,true),y(MINDEX(5,1));
	x.randomize();
	y.randomize();
	MachineLearning::TrainingDataset td1 = {x,y},td2;
	MachineLearning::save(td1,"test/saves/td.td");
	MachineLearning::load(td2,"test/saves/td.td");
	TEST_RETURN_FUNC(td1,==,td2);
}

void SaveLoadTest::Net() {
	PRINT_FUNC();
	MachineLearning::Net n1(MachineLearning::NetDef({5,4,3,2,1})), n2;
	TEST_RETURN_FUNC(MachineLearning::save(n1,"test/saves/testnet.nn"),==,true);
	TEST_RETURN_FUNC(MachineLearning::load(n2,"test/saves/testnet.nn"),==,true);
	TEST_RETURN_FUNC(n1.get_activation_function_list(),==,n2.get_activation_function_list());
	TEST_RETURN_FUNC(n1,==,n2);
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
									SaveLoadTest::test_list<__type__>((std::list<__type__>) __var__);

void SaveLoadTest::list() {
	PRINT_FUNC();
	MachineLearning::Net n(MachineLearning::NetDef({5,4,3,2,1}));
	TEST_LIST(MachineLearning::LayerParams,n);
	TEST_LIST(MachineLearning::ActivationFunction,n.get_activation_function_list());
	// SaveLoadTest::test_list<MachineLearning::LayerParams>((std::list<MachineLearning::LayerParams>)n);
	// SaveLoadTest::test_list<MachineLearning::ActivationFunction>(n.get_activation_function_list());
}

void SaveLoadTest::execute_all_tests() {
	SaveLoadTest::list();
	SaveLoadTest::LayerParams();
	SaveLoadTest::Matrix();
	SaveLoadTest::TrainingDataset();
	SaveLoadTest::ActivationFunction();
	SaveLoadTest::Net();
}