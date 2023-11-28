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
	if(m.get_num_rows() && m.get_num_cols()) {
		int j = 0;
		for (MachineLearning::mindex i = {0,0}; i.row < m.get_num_rows(); ++i.row) {
			for (i.col = 0; i.col < m.get_num_cols(); ++i.col) {
				m[i] = array[j];
				++j;
			}
		}
	}
}

bool MachineLearning::save(const MachineLearning::mindex& m,std::ofstream& file) {
	if(file.is_open()) {
		file.write((char*)(&m),sizeof(MachineLearning::mindex));
		return file.good();
	} else {
		return false;
	}
}

bool MachineLearning::load(MachineLearning::mindex& m, std::ifstream& file) {
	if(file.is_open()) {
		file.read((char*)&m,sizeof(MachineLearning::mindex));
		return file.good();
	} else {
		return false;
	}
}

bool MachineLearning::save(const LinearAlgebra::Matrix& m,std::ofstream& out_file) {
	if(!(out_file.is_open() && MachineLearning::save(m.size(),out_file))) {
		return false;
	}
	MachineLearning::uint N = m.get_num_rows()*m.get_num_cols();
	MachineLearning::scalar array[N];
	matrix_copy(m,array);
	out_file.write((char*)array,sizeof(MachineLearning::scalar)*N);
	return out_file.good();
}

bool MachineLearning::load(LinearAlgebra::Matrix& m, std::ifstream& in_file) {
	MachineLearning::mindex dims;
	if(!(in_file.is_open() && MachineLearning::load(dims,in_file))) {
		return false;
	}
	m.resize(dims);
	int N = dims.row*dims.col;
	MachineLearning::scalar array[N];
	in_file.read((char*)array,N*sizeof(MachineLearning::scalar));
	array_copy(array,m);
	return in_file.good();
}

bool MachineLearning::save(const MachineLearning::LayerParams& lp, std::ofstream& out_file) {
	if(out_file.is_open()) {
		return MachineLearning::save(lp.weights,out_file) && MachineLearning::save(lp.biases,out_file);
	} else {
		return false;
	}
}

bool MachineLearning::load(MachineLearning::LayerParams& lp, std::ifstream& in_file) {
	if(in_file.is_open()) {
		LinearAlgebra::Matrix weights, biases;
		return (MachineLearning::load(lp.weights,in_file) && MachineLearning::load(lp.biases,in_file));
	} else {
		return false;
	}
}

// NOTE: This function is not yet going to include the ability to save the ActivationFunction list
bool MachineLearning::save(const MachineLearning::Net& n, std::ofstream& out_file) {
	if(out_file.is_open()) {
		// LayerParams
		bool is_good = MachineLearning::save_list((std::list<MachineLearning::LayerParams>)n,out_file);
		// ActivationFunctions
		is_good = MachineLearning::save_list(n.get_activation_function_list(),out_file) && is_good;
		// Learning rate
		is_good = save(n.learning_rate,out_file) && is_good;
		// Training data
		is_good = save(n.get_training_data(),out_file) && is_good;
		return is_good;
	}
	return false;
}

bool MachineLearning::load(MachineLearning::Net& n, std::ifstream& in_file) {
	bool is_good = in_file.is_open();
	if(is_good) {
		std::list<MachineLearning::LayerParams> lpl;
		std::list<MachineLearning::ActivationFunction> afl;
		// LayerParams
		is_good = MachineLearning::load<MachineLearning::LayerParams>(lpl,in_file) && is_good;
		// ActivationFunctions
		is_good = MachineLearning::load_list<MachineLearning::ActivationFunction>(afl,in_file) && is_good;
		n = MachineLearning::Net(lpl,afl);
		// Learning rate
		is_good = load(n.learning_rate,in_file) && is_good;
		MachineLearning::TrainingDataset td;
		// TrainingDatset
		is_good = load(td,in_file) && is_good;
		n.load_training_data(td);
	}
	return is_good;
}

bool MachineLearning::save(MachineLearning::uint num, std::ofstream& out_file) {
	if(out_file.is_open()) {
		out_file.write((char*)(&num),sizeof(MachineLearning::uint));
		return out_file.good();
	} else {
		return false;
	}
}

bool MachineLearning::load(MachineLearning::uint& num, std::ifstream& in_file) {
	if(in_file.is_open()) {
		in_file.read((char*)(&num),sizeof(MachineLearning::uint));
		return in_file.good();
	} else {
		return false;
	}
}

bool MachineLearning::load(MachineLearning::scalar& num, std::ifstream& in_file) {
	if(in_file.is_open()) {
		in_file.read((char*)(&num),sizeof(MachineLearning::scalar));
		return in_file.good();
	} else {
		return false;
	}
}

bool MachineLearning::save(MachineLearning::scalar num, std::ofstream& out_file) {
	if(out_file.is_open()) {
		out_file.write((char*)(&num),sizeof(MachineLearning::scalar));
		return out_file.good();
	} else {
		return false;
	}
}

/**
 * @brief Temp function used to allow compilation
 */
bool MachineLearning::save(const MachineLearning::ActivationFunction& af, std::ofstream& out_file){
	if(out_file.is_open()) {
		func_sym sym = af.get_sym();
		out_file.write(&sym,sizeof(func_sym));
		return out_file.good();
	} else {
		return false;
	}
}

/**
 * @brief Temp function used to allow compilation
 */
bool MachineLearning::load(MachineLearning::ActivationFunction& af,std::ifstream& in_file){
	if(in_file.is_open()) {
		func_sym fs;
		in_file.read(&fs,1);
		af = MachineLearning::sym2ActFunc(fs);
	}
	return in_file.good();
}

bool MachineLearning::save(const MachineLearning::TrainingDataset& td, std::ofstream& out_file) {
	if (out_file.is_open()) {
		return MachineLearning::save(td.x,out_file) && MachineLearning::save(td.y,out_file);
	} else {
		return false;
	}
	
}

bool MachineLearning::load(MachineLearning::TrainingDataset& td, std::ifstream& in_file) {
	if (in_file.is_open()) {
		return MachineLearning::load(td.x,in_file) && MachineLearning::load(td.y,in_file);
	} else {
		return false;
	}
}