#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
/*Designation of Threshold function*/
enum Type{Sigmoidal , Linear};

/*Backpropagation network definition*/
struct BPN_CUDA{
	double* weight;							/*Weight of the network*/
	double* bias;							/*Bias value of each node in the network*/
	double* z_val;							/*Threshold output at each node*/
	double* a_val;							/*Weighted sum at the input side of the node, with bias added*/
	double* delta;							/*Delta value at each node(Required for weight and bias update)*/
	Type* type;								/*Type of threshold function at each level*/
	int noLevels;							/*Number of levels in the network*/
	int *nodeSize;							/*Size of levels(number of nodes in a level)*/
	int noNodes;							/*Total number of nodes in the network*/
	int noWeight;							/*Total number of weights in the network*/
	double training_rate;					/*Network training rate*/
};

/*Threshold and Differential Threshold functions*/
__device__ double computeFunc(double , Type);
__device__ double computeDiff(double , Type);


/*CUDA Kernel functions concerning the forward and reverse propagation of values within each level*/
/*The parameter with a 'curr' or no suffix indicates that its location is in the current node
The parameter having 'prev' suffix indicates its location to be in the previous(or current + 1) level
The parameter having 'next' suffix indicates its location to be in the next(or current - 1) level
The two dimensional network parameters like a_val , z_val , etc... are flattened to one-dimensional form for convenience in CUDA computation.
The two dimensional form of such parameters are for example: a_val[level_no][node_no]. Same applies for other 2D parameters
The three dimensional network weight parameter is also flattened to a one-dimensional form
Its non-flattened form is weight[level_number][node number of the current level][node number of the previous level]
*/
__global__ void forward_propagate_input(double* z_curr , double* bias , int size);
__global__ void forward_propagate_level(double* a_curr , double* z_curr , double* weight , double* z_prev , double* bias_curr , int prev , int next , Type t);
__global__ void reverse_propagate_output(double *delta_curr , double* z_curr , double* target , double* a_curr , int size , Type t);
__global__ void reverse_propagate_level(double *delta_curr , double *delta_next , double* weight_next , double* a_curr , Type t , int size_next , int size);

/*CUDA Kernel functions that updates the weight and bias values of the network*/
__global__ void weight_update(double* weight , double* delta_curr , double* z_prev , int size , int size_prev , double rate);
__global__ void bias_update(double* delta_curr , double* bias , int size , double rate);

void initialize(BPN_CUDA* network , int* noNodes , int levels , Type* type , double rate);

void forward(BPN_CUDA* network);
double reverse(BPN_CUDA* network , double* target , int size);
void weight_bias_update(BPN_CUDA* network , double rate);
void copyBPNinput(BPN_CUDA* network , double *input);

double train(BPN_CUDA* network , double* input , double* output , int dataset_no , int input_size , int output_size);
