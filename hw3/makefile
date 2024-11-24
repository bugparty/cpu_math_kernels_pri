CC = gcc
CXX = g++
CFLAGS = -I /home/act-software/mpich-3.2.1/include  -march=native -O3
CXXFLAGS =  -I /home/act-software/mpich-3.2.1/include -march=native -O3
LD_FLAGS = -L /home/act-software/mpich-3.2.1/lib -lmpi -lm -march=native -O3
SRCS = $(wildcard *.c)
OBJS = $(SRCS:.c=.o)

CXX_SRCS = $(wildcard *.cpp)
CXX_OBJS = $(CXX_SRCS:.cpp=.o)

TARGET = main

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS)  $(CXX_OBJS)
	$(CXX) $(OBJS)  $(CXX_OBJS)  $(LD_FLAGS) -o $(TARGET)

clean:
	rm -f $(C_OBJS) $(CXX_OBJS) $(TARGET)

main3: func_call.c include.h main.c makefile sieve0.c sieve1.c sieve2.cpp sieve3.c
	rm -f main
	gcc main.c -o main -I /home/act-software/mpich-3.2.1/include -L /home/act-software/mpich-3.2.1/lib -lmpi -lm -march=native -O3

