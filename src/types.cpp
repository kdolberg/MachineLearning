#include "types.h"

bool operator==(MachineLearning::TrainingDataset a, MachineLearning::TrainingDataset b) {
	return (a.x == b.x) && (a.y == b.y);
}