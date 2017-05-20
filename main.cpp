//#define USE_CUDA

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#define CUDA_HEADER
#include <stdlib.h>

#ifdef USE_CUDA
	#include "bpn_cuda.h"
#endif

#ifndef USE_CUDA
	#include "bpn.h"
#endif

#include "img_dataset_creator.h"
#include "mnist.h"

#define DEFAULT_NODE_COUNT 4000
#define DEFAULT_LAYER_COUNT 8

int main(int argc , char ** argv){

	int d_num , d_rows , d_cols , d_numl;
	double *dataset , *label;

	int n_count = DEFAULT_NODE_COUNT , n_layers = DEFAULT_LAYER_COUNT;

	if(argc > 1)
		n_count = atoi(argv[1]); 
	
	if(argc > 2)
		n_layers = atoi(argv[2]);


	Type *type = new Type[n_layers];
	int *layers = new int[n_layers];

	

	dataset = ReadMNISTIMAGE("train-images.idx3-ubyte" , d_num , d_rows , d_cols);
	label   = ReadMNISTLABEL("train-labels.idx1-ubyte" , d_numl);

	if(dataset == NULL || label == NULL){
		printf("Fatal: Couldn't read a MNIST file\n");
		exit(1);
	}

	if(d_num != d_numl){
		printf("Fatal: Image and label size didn't match.\n");
		exit(1);
	}


	int lambda = 2 * n_count / n_layers;

	if(lambda <= d_rows * d_cols){
		printf("Fatal: Input vector too large for the given node count.\n");
		exit(1);
	}

	for(int i = 0 ; i < n_layers ; i ++){
		if(i % 4 == 0)
			layers[i] = d_rows * d_cols;
		else if(i % 4 == 1)
			layers[i] = lambda - 1;
		else if(i % 4 == 2)
			layers[i] = lambda - d_rows * d_cols;
		else
			layers[i] = 1;
	
		type[i] = Linear;
	}
	
	double rate;

#ifndef USE_CUDA
	BPN* network;
	network = new BPN;
	rate = 0.0001;  /*Don't know the problem, but the network diverges for 
				large training_rate when CPU is used, but converges too 
				slow for small training_rate when CUDA is used*/

#else
	BPN_CUDA* network;
	network = new BPN_CUDA;
	rate = 0.0001;
#endif

	initialize(network , layers , n_layers , type , rate);
	
	clock_t begin = clock();
	int count = train(network , dataset , label , d_num , d_rows * d_cols , 1 , 5);
	clock_t end = clock();

	double time_spent = (double)(end - begin) / (CLOCKS_PER_SEC * count) * 1000;
	
	printf("Network Information:\n");
	printf("Number of Nodes : %d \n Number of Weight connections : %d \n" , network->noNodes , network->noWeight);
	printf("Time Spent : %f" , time_spent);
	


	return 0;
}

