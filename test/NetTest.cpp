#include "NetTest.h"

typedef struct NetUintIter {
	std::vector<MachineLearning::uint>::const_iterator def;
	MachineLearning::Net::const_iterator n;
	NetUintIter& operator++() {
		++this->def;
		++this->n;
		return (*this);
	}
	bool operator==(const NetUintIter& iter) {
		if((iter.n)==(this->n)) {
			assert((std::next(iter.def)==this->def) || (iter.def==std::next(this->def)));
			return true;
		} else {
			return false;
		}
	}
} NetUintIter;

#define H 0.001f

LinearAlgebra::scalar_t numerical_ddx(MachineLearning::Net& n,const MachineLearning::TrainingDataset& td, MachineLearning::scalar_t * wb_ptr) {
	LinearAlgebra::scalar_t E_avg1 = MachineLearning::error_avg(n.forward_propagate(td.x),td.y);
	LinearAlgebra::scalar_t tmp = (*wb_ptr);
	(*wb_ptr) += H;
	LinearAlgebra::scalar_t E_avg2 = MachineLearning::error_avg(n.forward_propagate(td.x),td.y);
	(*wb_ptr) = tmp;
	return ((E_avg2-E_avg1)/H);
}

MachineLearning::Gradient NetTest::numerical_gradient(MachineLearning::Net n, MachineLearning::TrainingDataset td) {
	MachineLearning::Gradient ret;
	for (auto i = n.std::list<MachineLearning::LayerStruct>::begin(); i != n.std::list<MachineLearning::LayerStruct>::end(); ++i) {
		LinearAlgebra::Matrix w_curr(i->params.weights.size()), b_curr(i->params.biases.size());
		MachineLearning::LayerParams curr = {w_curr,b_curr};
		for (LinearAlgebra::mindex_t j = {0,0}; j.row < i->params.weights.get_num_rows(); ++j.row) {
			LinearAlgebra::mindex_t k = MINDEX(j.row,0);
			curr.biases[k] = numerical_ddx(n,td,&(i->params.biases[k]));
			for (j.col = 0; j.col < i->params.weights.get_num_cols(); ++j.col) {
				curr.weights[j] = numerical_ddx(n,td,&(i->params.weights[j]));
			}
		}
		ret.push_back(curr);
	}
	return ret;
}

bool NetTest::Net_constructor_gives_correct_num_inputs() {
	std::vector<MachineLearning::uint> def = {5,4,3,2,1};
	MachineLearning::Net n(def);
	bool all_nums_correct = true;
	for (NetUintIter i = {def.cbegin(),n.cbegin()}; i != (NetUintIter){def.cend(),n.cend()}; ++i) {
		all_nums_correct = all_nums_correct && (i.n->params.get_num_inputs()==*(i.def));
	}
	return all_nums_correct;
}

void NetTest::forward_propagate() {
	std::vector<MachineLearning::uint> def = {2,2,1};
	MachineLearning::Net n(def,1.0f);
	MachineLearning::TrainingDataset td = {(LinearAlgebra::Matrix){{1},{1}},(LinearAlgebra::Matrix){{7}}};
	TEST_RETURN_FUNC(n.forward_propagate(td.x),==,td.y);
}

void NetTest::calculate_gradient() {
	MachineLearning::NetDef def = {2,1};
	MachineLearning::Net n(def,1.0f);
	LinearAlgebra::Matrix x = {{1,2,3,4,5},{1,2,3,4,5}};
	LinearAlgebra::Matrix y = {{1,2,3,4,5}};
	MachineLearning::TrainingDataset td = {x,y};
	TEST_RETURN_FUNC(n.calculate_gradient(td),==,numerical_gradient(n,td));
}

void NetTest::execute_all_tests() {
	TEST_RETURN_FUNC(Net_constructor_gives_correct_num_inputs(),==,true);
	forward_propagate();
	calculate_gradient();
}