#include "machine_learning.h"
#include "SaveLoadTest.h"

void SaveLoadTest::LayerParams() {
	LinearAlgebra::Matrix w(5,true),b(MINDEX(5,1));
	b[MINDEX(0,0)] = 1;
	MachineLearning::LayerParams lp1 = {w,b};
	save(lp1,"test/saves/1.lp");
	MachineLearning::LayerParams lp2;
	load(lp2,"test/saves/1.lp");
	TEST_RETURN_FUNC(lp1,==,lp2);
}

void SaveLoadTest::Matrix() {
	LinearAlgebra::Matrix m1(5,true), m2;
	save(m1,"test/saves/m.mat");
	load(m2,"test/saves/m.mat");
	TEST_RETURN_FUNC(m1,==,m2);
}

void SaveLoadTest::TrainingDataset() {
	LinearAlgebra::Matrix x(5,true),y(MINDEX(5,1));
	x.randomize();
	y.randomize();
	MachineLearning::TrainingDataset td1 = {x,y},td2;
	save(td1,"test/saves/td.td");
	load(td2,"test/saves/td.td");
	TEST_RETURN_FUNC(td1,==,td2);
}

void SaveLoadTest::execute_all_tests() {
	SaveLoadTest::LayerParams();
	SaveLoadTest::Matrix();
	SaveLoadTest::TrainingDataset();
}