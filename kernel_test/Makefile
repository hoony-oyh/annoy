CC = gcc
CFLAGS = -O3 -std=c++11 -mavx -msse2

all:
	$(CC) $(CFLAGS) $(OBJS) dot.cpp  -o dot.x -march=native
run:
	./dot.x

clean:
	rm -f *.x *~ *.o