#ifndef __INCLUDE__HOST
#define __INCLUDE__HOST
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#endif

#include <fstream>


int reverseInt(int i);

double** ReadMNISTIMAGE(char *filename , int& num_image , int& size_image_rows , int& size_image_cols );
 
double* ReadMNISTLABEL(char* filename , int& size);