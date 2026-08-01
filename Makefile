CXX = clang++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -I./include

# 1. Directories
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj

# 2. Targets
TARGET = $(BUILD_DIR)/engine
TEST_TARGET = $(BUILD_DIR)/test_runner

# 3. Source files (everything in src/ except main.cpp)
CORE_SRCS = $(filter-out src/main.cpp, $(wildcard src/*.cpp))

# 4. Map .cpp files to .o files inside the obj/ folder
# e.g., src/tokenizer.cpp -> build/obj/tokenizer.o
CORE_OBJS = $(patsubst src/%.cpp, $(OBJ_DIR)/%.o, $(CORE_SRCS))
MAIN_OBJ = $(OBJ_DIR)/main.o
TEST_OBJ = $(OBJ_DIR)/tests.o

all: $(TARGET)

# 5. Engine Linking Phase
# $^ is a Make variable that means "all prerequisites" (CORE_OBJS and MAIN_OBJ)
# $@ means "the target name" (TARGET)
$(TARGET): $(CORE_OBJS) $(MAIN_OBJ)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

# 6. Run Phase
run: $(TARGET)
	./$(TARGET)

# 6. Test Linking Phase
test: $(CORE_OBJS) $(TEST_OBJ)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $(TEST_TARGET)
	./$(TEST_TARGET)

# 7. Compile src/ files into build/obj/
$(OBJ_DIR)/%.o: src/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 8. Compile root-level files into build/obj/
$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 9. Compile test/ files into build/obj/
$(OBJ_DIR)/%.o: test/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 10. Cleanup is now just deleting the build directory!
clean:
	rm -rf $(BUILD_DIR)