# Git version
# GIT_VERSION := "$(shell git describe --abbrev=4 --dirty --always --tags)"

# compiler
CXX = g++

# compiler flags
SCALAR_TYPE=float
DEFINE_SCALAR_TYPE_MACRO = -DSCALAR_TYPE=$(SCALAR_TYPE)
FLAGS = -DNOFLAGS -UUNIT_TEST
CXXFLAGS = -std=c++23 $(DEFINE_SCALAR_TYPE_MACRO) $(FLAGS) -Wall -g -O0

# directory of header files
INCLUDES = -I. -I./../Utilities -I./../Utilities/src -I./../Utilities/include -I./include

# sources and objects
LINALG_DIR = ../Utilities
LINALG_OBJS = ../Utilities/obj/la_basic_types.la ../Utilities/obj/la_matrix.la ../Utilities/obj/la_matrix_like.la ../Utilities/obj/la_vector.la ../Utilities/obj/la_vector_overloads.la
ML_OBJ = obj/main.ml obj/activation_function.ml
OBJECTS = $(ML_OBJ) $(LINALG_OBJS)

# target executable
TARGET = a

$(TARGET): $(ML_OBJ)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) $(OBJECTS)

linalg:
	make -C $(LINALG_DIR) obj/la_basic_types.la obj/la_matrix.la obj/la_matrix_like.la obj/la_vector.la obj/la_vector_overloads.la

obj/%.ml: src/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(ML_OBJ) $(TARGET)

clean_linalg:
	rm $(LINALG_OBJS)

.PHONY: clean clean_linalg linalg