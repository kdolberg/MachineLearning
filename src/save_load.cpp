#include "save_load.h"

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

void matrix_copy(const LinearAlgebra::Matrix& m, MachineLearning::scalar * array) {
	int j = 0;
	for (MachineLearning::mindex i = {0,0}; i.row < m.get_num_rows(); ++i.row) {
		for (i.col = 0; i.col < m.get_num_cols(); ++i.col) {
			array[j] = m[i];
			++j;
		}
	}
}

void array_copy(const MachineLearning::scalar * array,LinearAlgebra::Matrix& m) {
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

template <typename T>
bool save(const T& t, const char * filename) {
	std::ofstream out_file;
	out_file.open(filename,std::ios::out | std::ios::binary);
	save(t,out_file);
	out_file.close();
	return out_file.good();
}

template <typename T>
bool load(T& t, const char * filename) {
	std::ifstream in_file;
	in_file.open(filename,std::ios::in | std::ios::binary);
	load(t,in_file);
	in_file.close();
	return in_file.good();
}

void test() {
	LinearAlgebra::Matrix m1(5,true),m2;
	std::cout << m1 << std::endl;
	std::cout << save(m1,"MyMatrix.mat") << std::endl;
	std::cout << load(m2,"MyMatrix.mat") << std::endl;
	std::cout << m2 << std::endl;
	std::cout << (m1==m2) << std::endl;
}