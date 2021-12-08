####### Compiler, tools and options

CC            = gcc
CXX           = g++-4.4
CFLAGS        = -m64 -pipe -O2 -Wno-unused-but-set-parameter -Wall
CXXFLAGS      = -m64 -pipe -O2 -Wno-unused-but-set-parameter -Wall
OPENCL        = /opt/AMDAPP
CLFFT         = /libclFFT.2.12.2.dylib
INC_OPENCL    = -I$(OPENCL)/include
INC_CLFFT     = -I$(CLFFT)/src/include
INCPATH       = $(INC_OPENCL) $(INC_CLFFT)
LIB_FFTW      = -lfftw3
LIB_CLFFT     = /usr/local/lib/libclFFT.2.12.2.dylib -lclFFT -framework OpenCL
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
