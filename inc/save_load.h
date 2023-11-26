#ifndef SAVE_LOAD_H
#define SAVE_LOAD_H

#include <fstream>

#include "linear_algebra.h"
#include "net.h"
#include "layer.h"
#include "misc_overloads.h"

namespace MachineLearning {
	bool save(const MachineLearning::mindex&,std::ofstream&);
	bool save(const LinearAlgebra::Matrix&,std::ofstream&);
	bool save(const MachineLearning::LayerParams&,std::ofstream&);
	bool save(const MachineLearning::Net&,std::ofstream&);
	bool save(const MachineLearning::ActivationFunction&, std::ofstream&);
	bool save(MachineLearning::uint, std::ofstream&);
	bool save(const MachineLearning::TrainingDataset&, std::ofstream&);

	bool load(MachineLearning::mindex&,std::ifstream&);
	bool load(LinearAlgebra::Matrix&,std::ifstream&);
	bool load(MachineLearning::LayerParams&,std::ifstream&);
	bool load(MachineLearning::Net&,std::ifstream&);
	bool load(MachineLearning::ActivationFunction&,std::ifstream&);
	bool load(MachineLearning::uint&, std::ifstream&);
	bool load(MachineLearning::TrainingDataset&, std::ifstream&);

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

	template <typename T>
	bool save_list(const std::list<T>& l, std::ofstream& out_file) {
		assert(!(l.empty()));
		bool is_good = out_file.is_open();
		if(is_good) {
			is_good = save(l.size(),out_file) && is_good;
			for (auto i = l.cbegin(); i != l.cend(); ++i) {
				is_good = save(*i,out_file) && is_good;
			}
		}
		return is_good;
	}

	template <typename T>
	bool load_list(std::list<T>& l, std::ifstream& in_file) {
		assert(l.empty());
		bool is_good = in_file.is_open();
		if(is_good) {
			MachineLearning::uint length;
			is_good = load(length,in_file) && is_good;
			for (MachineLearning::uint i = 0; i < length; ++i) {
				T tmp;
				is_good = load(tmp,in_file) && is_good;
				l.push_back(tmp);
			}
		}
		return is_good;
	}

	template <typename T>
	bool save(const std::list<T>& l, std::ofstream& out_file) {
		return save_list<T>(l,out_file);
	}

	template <typename T>
	bool load(std::list<T>& l, std::ifstream& in_file) {
		assert(l.empty());
		return load_list<T>(l,in_file);
	}
} // MachineLearning

#endif // SAVE_LOAD_H