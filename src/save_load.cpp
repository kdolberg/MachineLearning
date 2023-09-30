#include <stdint.h>
#include "save_load.h"

static void matrix_copy(const LinearAlgebra::Matrix& m, MachineLearning::scalar * array) {
	int j = 0;
	for (MachineLearning::mindex i = {0,0}; i.row < m.get_num_rows(); ++i.row) {
		for (i.col = 0; i.col < m.get_num_cols(); ++i.col) {
			array[j] = m[i];
			++j;
		}
	}
}

static void array_copy(const MachineLearning::scalar * array,LinearAlgebra::Matrix& m) {
	assert(m.get_num_rows());
	assert(m.get_num_cols());
	int j = 0;
	for (MachineLearning::mindex i = {0,0}; i.row < m.get_num_rows(); ++i.row) {
		for (i.col = 0; i.col < m.get_num_cols(); ++i.col) {
			m[i] = array[j];
			++j;
		}
	}
}

bool save(const MachineLearning::mindex& m,std::ofstream& file) {
	if(file.is_open()) {
		file.write((char*)(&m),sizeof(MachineLearning::mindex));
		return file.good();
	} else {
		return false;
	}
}

bool load(MachineLearning::mindex& m, std::ifstream& file) {
	if(file.is_open()) {
		file.read((char*)&m,sizeof(MachineLearning::mindex));
		return file.good();
	} else {
		return false;
	}
}

bool save(const LinearAlgebra::Matrix& m,std::ofstream& out_file) {
	if(!(out_file.is_open() && save(m.size(),out_file))) {
		return false;
	}
	MachineLearning::uint N = m.get_num_rows()*m.get_num_cols();
	MachineLearning::scalar array[N];
	matrix_copy(m,array);
	out_file.write((char*)array,sizeof(MachineLearning::scalar)*N);
	return out_file.good();
}

bool load(LinearAlgebra::Matrix& m, std::ifstream& in_file) {
	MachineLearning::mindex dims;
	if(!(in_file.is_open() && load(dims,in_file))) {
		return false;
	}
	m.resize(dims);
	int N = dims.row*dims.col;
	MachineLearning::scalar array[N];
	in_file.read((char*)array,N*sizeof(MachineLearning::scalar));
	array_copy(array,m);
	return in_file.good();
}

bool save(const MachineLearning::LayerParams& lp, std::ofstream& out_file) {
	if(out_file.is_open()) {
		return save(lp.weights,out_file) && save(lp.biases,out_file);
	} else {
		return false;
	}
}

bool load(MachineLearning::LayerParams& lp, std::ifstream& in_file) {
	if(in_file.is_open()) {
		LinearAlgebra::Matrix weights, biases;
		return (load(lp.weights,in_file) && load(lp.biases,in_file));
	} else {
		return false;
	}
}

// NOTE: This function is not yet going to include the ability to save the ActivationFunction list
bool save(const MachineLearning::Net& n, std::ofstream& out_file) {
	if(out_file.is_open()) {
		bool is_good = save_list((std::list<MachineLearning::LayerParams>)n,out_file);
		// is_good = save_list(n.get_activation_function_list(),out_file) && is_good;
		return is_good;
	}
	return false;
}

bool load(MachineLearning::Net& n, std::ifstream& in_file) {
	bool is_good = in_file.is_open();
	if(is_good) {
		is_good = load_list(n,in_file) && is_good;
		std::list<MachineLearning::ActivationFunction> af_list;
		is_good = load_list(af_list,in_file) && is_good;
	}
	return is_good;
}

bool save(MachineLearning::uint num, std::ofstream& out_file) {
	if(out_file.is_open()) {
		out_file.write((char*)(&num),sizeof(MachineLearning::uint));
		return out_file.good();
	} else {
		return false;
	}
}

bool load(MachineLearning::uint& num, std::ifstream& in_file) {
	if(in_file.is_open()) {
		in_file.read((char*)(&num),sizeof(MachineLearning::uint));
		return in_file.good();
	} else {
		return false;
	}
}

/**
 * @brief Temp function used to allow compilation
 */
bool save(const MachineLearning::ActivationFunction&, std::ofstream&){
	return true;
}
/**
 * @brief Temp function used to allow compilation
 */
bool load(MachineLearning::ActivationFunction&,std::ifstream&){
	return true;
}