#include "NetPerformance.h"

LinearAlgebra::Matrix NetTest::Performance::XOR(LinearAlgebra::Matrix& a) {
	assert(a.get_num_rows()==2);
	assert(a.get_num_cols() > 0);
	LinearAlgebra::mindex_t dims = a.size();
	dims.row = 1;
	LinearAlgebra::Matrix ret(dims);

	for (LinearAlgebra::mindex_t i = {0,0}; i.col < a.get_num_cols(); ++i.col) {
		ret[i] = NetTest::Performance::XOR(a[MINDEX(0,i.col)],a[MINDEX(1,i.col)]);
	}

	return ret;
}

MachineLearning::TrainingDataset NetTest::Performance::xor_dataset() {
	LinearAlgebra::Matrix _x = {{0,0},
								{0,1},
								{1,0},
								{1,1}};
	LinearAlgebra::Matrix x = LinearAlgebra::transpose(_x);
	LinearAlgebra::Matrix y = NetTest::Performance::XOR(x);
	MachineLearning::TrainingDataset ret = {x,y};
	return ret;
}

void NetTest::Performance::learn_xor() {
	MachineLearning::TrainingDataset td = xor_dataset();

	MachineLearning::NetDef def = {2,100,100,50,1};
	MachineLearning::Net n(def,true);

	for (int i = 0; i < 10; ++i) {
		PRINT_VAR(n.learn(td));
	}
}

void NetTest::Performance::execute_all_tests() {
	NetTest::Performance::learn_xor();
}