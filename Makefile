COMPILER=gcc
CFLAGS=-c -Wall -O3
LINKFLAGS=

all: arm_mul_demo

arm_mul_demo: main.o
	$(COMPILER) main.o $(LINKFLAGS) -o arm_mul_demo

main.o: main.c
	$(COMPILER) $(CFLAGS) main.c

clean:
	rm -rf *.o arm_mul_demo

