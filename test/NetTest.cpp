#include "NetTest.h"

bool NetTest::Net_constructor_gives_correct_num_inputs() {
	std::vector<MachineLearning::uint> def = {5,4,3,2,1};
	MachineLearning::Net n(def);
	bool all_nums_correct = true;
	for (NetTest::NetUintIter i = {def.cbegin(),n.cbegin()}; i != (NetTest::NetUintIter){def.cend(),n.cend()}; ++i) {
		std::cout << "# inputs: " << i.n->params.get_num_inputs() << std::endl;
		std::cout << (i.n)->params << std::endl;
		all_nums_correct = all_nums_correct && (i.n->params.get_num_inputs()==*(i.def));
	}
	return all_nums_correct;
}

void NetTest::forward_propagation_output() {
	std::vector<MachineLearning::uint> def = {5,4,3,2,1};
	MachineLearning::Net n(def,1.0f);
}

void NetTest::execute_all_tests() {
	TEST_RETURN_FUNC(Net_constructor_gives_correct_num_inputs(),==,true);
}