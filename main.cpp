#include "bpn_cuda.h"

int main(){
	double input[8] = {0 , 0 , 0 , 1 , 1 , 0 , 1 , 1};
	double output[4] = {0 , 1 , 1 , 0};
	Type type[3] = {Linear , Sigmoidal , Linear};
	int noNodes[3] = {1 , 3 , 2};
	BPN_CUDA* network;
	network = new BPN_CUDA;
	initialize(network , noNodes , 3 , type , 4);
	train(network , input , output , 4 , 2 , 1);
}