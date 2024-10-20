CC = gcc
CXX= g++
OPT ?= -O0
CXXFLAGS = -Wall -march=native -g $(OPT)   
CFLAGS = -Wall   -march=native -g $(OPT)
main: dgemm0.c dgemm1.c dgemm2.c dgemm3.c dgemm7.c \
	dgemm6_ijk.c dgemm6_ikj.c dgemm6_jik.c dgemm6_jki.c dgemm6_kij.c dgemm6_kji.c \
	main.c makefile include.c
	rm -f main
	$(CC) $(CFLAGS) main.c -o main
tester : dgemm0.c dgemm1.c dgemm2.c dgemm3.c dgemm7.c \
	dgemm6_ijk.c dgemm6_ikj.c dgemm6_jik.c dgemm6_jki.c dgemm6_kij.c dgemm6_kji.c \
	tester.c makefile
	rm -f tester
	$(CC) $(CFLAGS) tester.c -o tester

CXX_SOURCES = bench.cpp
C_SOURCES = dgemm.c
CXX_OBJECTS = $(CXX_SOURCES:.cpp=.o)
C_OBJECTS = $(C_SOURCES:.c=.o)

TARGET = bench

$(TARGET): $(CXX_OBJECTS) $(C_OBJECTS)
	$(CXX) $(CXXFLAGS) $(CXX_OBJECTS) $(C_OBJECTS) -o $(TARGET) 

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
.PHONY: clean
clean:
	rm -f main tester *.o
