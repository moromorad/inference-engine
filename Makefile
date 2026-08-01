CXX = clang++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -I./include

# -O3   : Level 3 Optimization (Crucial for fast AI math)
# -I... : Tells the compiler where to look for your .h files

TARGET = build/engine
SRC = src/main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)