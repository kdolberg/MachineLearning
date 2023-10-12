#include "main.h"

void save_load_test() {
	LinearAlgebra::Matrix w(5,true),b(MINDEX(5,1));
	b[MINDEX(0,0)] = 1;
	MachineLearning::LayerParams lp1 = {w,b};
	save(lp1,"1.lp");
	MachineLearning::LayerParams lp2;
	load(lp2,"1.lp");
	TEST_RETURN_FUNC(lp1,==,lp2);
}

int main(int argc, char const *argv[]) {
	NetTest::execute_all_tests();
	ActFuncTest::execute_all_tests();
	print_report_card();
	return 0;
}