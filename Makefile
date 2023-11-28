# Compiler
CXX = g++

LA_DIR = LinearAlgebra

# Local directories
SRC_DIR = src
INC_DIR = inc
OBJ_DIR = obj
DOC_DIR = doc
BIN_DIR = bin
TEST_DIR = test
MKDIRIFNEEDED = @mkdir -p $(@D)

# Compiler flags
SCALAR_TYPE=float
DEFINE_SCALAR_TYPE_MACRO = -DSCALAR_TYPE=$(SCALAR_TYPE)
FLAGS = -DNOFLAGS
CXXFLAGS = -std=c++23 $(DEFINE_SCALAR_TYPE_MACRO) $(FLAGS) -Wall -g -O2 -MMD -MP

# Directory of header files
OTHER_DIR = -I./../../UnitTest -I./../../UnitTest/$(INC_DIR) -I./../../confirm
INCLUDES = -I. -I./$(LA_DIR) -I./$(INC_DIR) $(OTHER_DIR)

# Sources and objects
LINALG_OBJ = $(OBJ_DIR)/la_basic_types.la $(OBJ_DIR)/la_matrix.la $(OBJ_DIR)/la_matrix_like.la $(OBJ_DIR)/la_vector.la $(OBJ_DIR)/la_vector_overloads.la
ML_OBJ = $(OBJ_DIR)/activation_function.ml $(OBJ_DIR)/layer.ml $(OBJ_DIR)/net.ml $(OBJ_DIR)/save_load.ml $(OBJ_DIR)/types.ml
ML_TEST_OBJ = $(OBJ_DIR)/main.ml_test $(OBJ_DIR)/NetTest.ml_test $(OBJ_DIR)/NetPerformance.ml_test $(OBJ_DIR)/ActFuncTest.ml_test $(OBJ_DIR)/SaveLoadTest.ml_test
UNIT_TEST_OBJ = $(OBJ_DIR)/test.unit_test
OBJECTS = $(ML_OBJ) $(LINALG_OBJ) $(ML_TEST_OBJ) $(UNIT_TEST_OBJ)

# Target executable
TARGET = $(BIN_DIR)/test

$(TARGET): $(OBJECTS) clean_saves
	$(MKDIRIFNEEDED)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJECTS) -o $(TARGET)

$(OBJ_DIR)/%.la: $(LA_DIR)/$(SRC_DIR)/%.cpp $(LA_DIR)/$(SRC_DIR)/%.h
	$(MKDIRIFNEEDED)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/%.ml: $(SRC_DIR)/%.cpp $(INC_DIR)/%.h
	$(MKDIRIFNEEDED)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/%.ml_test: $(TEST_DIR)/%.cpp $(TEST_DIR)/%.h
	$(MKDIRIFNEEDED)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/%.unit_test: ../../UnitTest/$(SRC_DIR)/%.cpp ../../UnitTest/$(INC_DIR)/%.h
	$(MKDIRIFNEEDED)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean: clean_saves
	rm -f $(ML_OBJ) $(ML_TEST_OBJ) $(UNIT_TEST_OBJ) $(TARGET).exe

clean_linalg:
	rm -f $(LINALG_OBJ)

clean_ml_test: clean_saves
	rm -f $(ML_TEST_OBJ)

linalg: $(LINALG_OBJ)

clean_all: clean clean_linalg

all: $(TARGET)

run: $(TARGET) clean_saves
	./$(TARGET).exe

debug: $(TARGET) clean_saves
	gdb $(TARGET).exe -x gdb_cmd

folders:
	mkdir --parents $(INC_DIR) $(OBJ_DIR) $(SRC_DIR) $(DOC_DIR) $(TEST_DIR) $(BIN_DIR)

clean_saves:
	rm -f test/saves/*

.PHONY: clean clean_linalg linalg clean_all run all folders clean_saves debug