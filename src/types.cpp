#include "types.h"

bool operator==(const MachineLearning::TrainingDataset& a, const MachineLearning::TrainingDataset& b) {
	return (a.x == b.x) && (a.y == b.y);
}