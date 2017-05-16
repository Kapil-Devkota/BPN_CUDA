//#define USE_CUDA

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CUDA_HEADER

#ifdef USE_CUDA
	#include "bpn_cuda.h"
#endif

#ifndef USE_CUDA
	#include "bpn.h"
#endif

#include "img_dataset_creator.h"

int main(){
	
	int dim = 40;
	double *input = new double[dim * dim * 5] , *ip = new double[dim * dim];

	for(int i = 1 ; i < 6 ; i ++){
		char file[20];
		sprintf(file , "image\\image%d.bmp" , i);
		createImageVector(file , dim , ip);
		memcpy(input + (i - 1) * dim * dim , ip , dim * dim * sizeof(double));
	}

	double output[5] = { -2 
						,-1
						, 0
						, 1
						, 2
					   };
	

/*	Input Size = 900
	Type type[25] = {Linear , Linear , Linear , Linear , Linear , Linear , Linear , Linear , Linear , Linear , Linear , Linear , Linear , Linear , Linear , Linear , Linear , Linear , Linear , Linear , Linear , Linear , Linear , Linear , Linear};
	int noNodes[25] = {1 , 5 , 10 , 17 , 20 , 25 , 42 , 60 , 95 , 120 , 195 , 250 , 320 , 380 , 400 , 450 , 500 , 550 , 600 , 650 , 700 , 750 , 820 , 850 , 900};
*/	
	Type type[16] = {Linear , Linear , Linear , Linear , Linear , Linear , Linear , Linear , Linear , Linear , Linear , Linear , Linear , Linear , Linear , Linear};
	int noNodes[16] = {1 , 15 , 50 , 105 , 250 , 340 , 400 , 550 , 650 , 710 , 800 , 904 , 1025 , 1200 , 1400 , dim * dim};
	int nL = 16;
	double rate;
	

#ifndef USE_CUDA
	BPN* network;
	network = new BPN;
	rate = 0.0002;  /*Don't know the problem, but the network diverges for 
				large training_rate when CPU is used, but converges too 
				slow for small training_rate when CUDA is used*/

#else
	BPN_CUDA* network;
	network = new BPN_CUDA;
	rate = 0.0001;
#endif


	initialize(network , noNodes , nL , type , rate);
	
	clock_t begin = clock();
	int count = train(network , input , output , 5 , dim * dim , 1);
	clock_t end = clock();

	double time_spent = (double)(end - begin) / (CLOCKS_PER_SEC * count) * 1000;
	
	printf("Network Information:\n");
	printf("Number of Nodes : %d \n Number of Weight connections : %d \n" , network->noNodes , network->noWeight);
	printf("Time Spent : %f" , time_spent);
	
	return 1;
}