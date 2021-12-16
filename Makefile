####### Compiler, tools and options

CC            = gcc
CXX           = g++-4.4
CFLAGS        = -m64 -pipe -O2 -Wno-unused-parameter -Wall
CXXFLAGS      = -m64 -pipe -O2 -Wno-unused-parameter -Wall
OPENCL        = /opt/AMDAPP
INC_OPENCL    = -I$(OPENCL)/include
INC_CLFFT     = -I../../2.12.2/include
INCPATH       = $(INC_OPENCL) $(INC_CLFFT)
LIB_FFTW      = -lfftw3
LIB_CLFFT     = -L/usr/local/lib -lclFFT -framework OpenCL
LIB_MATH      = -lm
DEL_FILE      = rm -f

####### Build rules

all: FFT_2D FFT_2D_OpenCL

# Classic FFT with fftw
FFT_2D: main.c
	$(CC) $(CFLAGS) $(LIB_FFTW) $(LIB_MATH) -o FFT_2D main.c

# OpenCL FFT
FFT_2D_OpenCL: main_OpenCL.c
	$(CC) $(CFLAGS) $(INCPATH) $(LIB_CLFFT) $(LIB_MATH) -o FFT_2D_OpenCL main_OpenCL.c

clean:
	$(DEL_FILE) FFT_2D FFT_2D_OpenCL
	$(DEL_FILE) *.o
