#ifndef SAVING_AND_LOADING_H
#define SAVING_AND_LOADING_H

#include <fstream>

#include "linear_algebra.h"
#include "net.h"
#include "layer.h"

bool save(const MachineLearning::mindex& m,std::ofstream& file);

bool save(const LinearAlgebra::Matrix&,std::ofstream& file);

void test();

#endif // SAVING_AND_LOADING_H