CC = gcc
LD = gcc
CFLAGS = -g -O3 -march=native -Wall -std=c99
INCLUDES=
LDFLAGS=-lm -lpthread
LDLIBS =
RM = /bin/rm -f
OBJS = LDLTdecomp.o matrix.o
EXEC = LDLTdecomp

all: $(EXEC)

$(EXEC): $(OBJS)
	$(LD) -o $(EXEC) $(OBJS) $(LDFLAGS) $(LDLIBS) $(INCLUDES)

LDLTdecomp.o: LDLTdecomp.c matrix.c matrix.h
	$(CC) $(CFLAGS) $(INCLUDES) matrix.c -c LDLTdecomp.c

matrix.o: matrix.c matrix.h
	$(CC) $(CFLAGS) $(INCLUDES) -c matrix.c

#utils.o: utils.c utils.h constants.h graphics.h
#	$(CC) $(CFLAGS) $(INCLUDES) -c utils.c

clean:
	$(RM) $(EXEC) $(OBJS)
