CC = gcc
CFLAGS = -O3 -std=c++11 -mavx -msse2

all: dot matrix_dot add scale fma clean

dot:
	$(CC) $(CFLAGS) $(OBJS) dot.cpp  -o dot.x -march=native
	./dot.x 1000 512

add:
	$(CC) $(CFLAGS) $(OBJS) add.cpp  -o add.x -march=native
	./add.x 1000 512

scale:
	$(CC) $(CFLAGS) $(OBJS) scale.cpp  -o scale.x -march=native
	./scale.x 1000 512

fma:
	$(CC) $(CFLAGS) $(OBJS) fma.cpp  -o fma.x -march=native
	./fma.x 1000 512

matrix_dot:
	$(CC) $(CFLAGS) $(OBJS) matrix_dot.cpp -o matrix_dot.x -march=native -mfma -lstdc++
	./matrix_dot.x 1000 512

clean:
	rm -f *.x *~ *.o