# Git version
# GIT_VERSION := "$(shell git describe --abbrev=4 --dirty --always --tags)"

# compiler
CXX = g++

# compiler flags
SCALAR_TYPE=float
DEFINE_SCALAR_TYPE_MACRO = -DSCALAR_TYPE=$(SCALAR_TYPE)
FLAGS = -DNOFLAGS
CXXFLAGS = -std=c++23 $(DEFINE_SCALAR_TYPE_MACRO) $(FLAGS) -Wall -g -O0

# directory of header files
INCLUDES = -I. -I./../Utilities -I./../Utilities/src -I./../Utilities/include

# sources and objects
LINALG_OBJS = ../Utilities/src/la_basic_types.la ../Utilities/src/la_matrix.la ../Utilities/src/la_matrix_like.la ../Utilities/src/la_vector.la ../Utilities/src/la_vector_overloads.la
ML_SRC = main.cpp activation_function.cpp
ML_OBJ = $(ML_SRC:.cpp=.ml)
OBJECTS = $(LINALG_OBJS) $(ML_OBJ)

# target executable
TARGET = a

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) $(OBJECTS)

%.la: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.ml: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(ML_OBJ) $(TARGET)

linalg:
	make $(LINALG_OBJS)

.PHONY: clean linalg