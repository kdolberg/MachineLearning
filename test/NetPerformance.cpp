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

	MachineLearning::NetDef def = {2,50,50,50,50,1};
	MachineLearning::Net n(def,true);

	for (auto j = n.afs.begin(); j != n.afs.end(); ++j) {
		std::cout << (*j) << std::endl;
	}

	MachineLearning::scalar error_prev = 1;
	MachineLearning::scalar error_curr = 2;

	for (int i = 0; (i < 10000) && (error_curr > 0.1) && (error_curr != error_prev); ++i) {
		std::cout << i << ": ";
		error_prev = error_curr;
		error_curr = n.learn(td);
		PRINT_VAR(error_curr);
	}

	PRINT_VAR(n());
}

void NetTest::Performance::execute_all_tests() {
	NetTest::Performance::learn_xor();
}