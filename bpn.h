#include "common.h"
namespace CPU{
struct BPN{
	double* weight;
	double* bias;
	double* z_val;
	double* a_val;
	double* delta;
	Type* type;
	int noLevels;
	int *nodeSize;
	int noNodes;
	int noWeight;
	double training_rate;
};


double computeFuncH(double , Type);
double computeDiffH(double , Type);


void getLevelNodes(BPN *network , int level , double** z , double** a , double** bias, double** delta , int* size);
void getLevelWeights(BPN* network , int level , double** weights , int* size , int* length);
void forward_propagate_level(int level , BPN *network , double* input);
void reverse_propagate_level(int level , BPN *network , double* target);
void weight_bias_update(BPN *network);
int train(BPN* network , double* input , double* output , int dataset_no , int input_size , int output_size , int total_iterations = -1);
double reverse(BPN* network , double* target);
void forward(BPN* network , double* input);
void initialize(BPN* network , int* noNodes , int levels , Type* type , double rate);
void returnOutput(BPN* network , double* input , double* output);
}