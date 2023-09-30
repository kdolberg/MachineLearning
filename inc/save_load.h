#ifndef SAVE_LOAD_H
#define SAVE_LOAD_H

#include <fstream>

#include "linear_algebra.h"
#include "net.h"
#include "layer.h"

bool save(const MachineLearning::mindex&,std::ofstream&);
bool save(const LinearAlgebra::Matrix&,std::ofstream&);
bool save(const MachineLearning::LayerParams&,std::ofstream&);
bool save(const MachineLearning::Net&,std::ofstream&);

bool load(MachineLearning::mindex&,std::ifstream&);
bool load(LinearAlgebra::Matrix&,std::ifstream&);
bool load(MachineLearning::LayerParams&,std::ifstream&);
bool load(MachineLearning::Net&,std::ifstream&);

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

#endif // SAVE_LOAD_H