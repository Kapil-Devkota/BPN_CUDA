
#include "bpn.h"

double computeFuncH(double x , Type t){
	if(t == Linear)
		return x;
	if(t == Sigmoidal)
		return 1 / (exp(-x) + 1);
	
}

double computeDiffH(double x , Type t){
	if(t == Linear)
		return 1;
	else{
		double sig = computeFuncH(x , t);
		return sig * (1 - sig);
	}
}


void getLevelNodes(BPN *network , int level , double** z , double** a , double** bias, double** delta , int* size){
	int start = 0;

	for(int i = 0 ; i < level ; i ++){
		start += network->nodeSize[i];
	}

	*size = network->nodeSize[level];

	if(z != NULL)
		*z = network->z_val + start;
	
	if(a != NULL)
		*a = network->a_val + start;

	if(bias != NULL)
		*bias = network->bias + start;

	if(delta != NULL)
		*delta = network->delta + start;

	return;
}

void getLevelWeights(BPN* network , int level , double** weights , int* size , int* length){
	if(level == network->noLevels - 1)
		return;

	int start = 0;
	for(int i = 0 ; i < level ; i ++){
		start += network->nodeSize[i] * network->nodeSize[i + 1];
	}

	*size = network->nodeSize[level + 1];
	*length = network->nodeSize[level];

	*weights = network->weight + start;
	return;
}

void forward_propagate_level(int level , BPN *network , double* input){
	if(level == network->noLevels - 1){//input layer
		if(input == NULL)
			return;
		
		double *z , *a , *bias;
		int size;

		getLevelNodes(network , level , &z , &a , &bias , NULL , &size);
		for(int i = 0 ; i < size ; i ++){
			a[i] = input[i];
			z[i] = a[i] + bias[i];
		}
		/*DEBUG
		printf("\nDEBUG:\n");
		for(int i = 0 ; i < size ; i ++)
			printf("%f \n" , z[i]);
		DEBUG COMPLETE*/

		return;
	}

	double *z_prev , *weight , *a_curr , *z_curr , *bias_curr;

	int size , sizePrev;
	
	getLevelWeights(network , level , &weight , &sizePrev , &size);

	getLevelNodes(network , level + 1 , &z_prev , NULL , NULL , NULL , &sizePrev);

	getLevelNodes(network , level , &z_curr , &a_curr , &bias_curr , NULL , &size);

	for(int i = 0 ; i < size ; i ++){
		a_curr[i] = 0;
		for(int j = 0 ; j < sizePrev ; j ++){
			a_curr[i] += *(weight + i * sizePrev + j) * z_prev[j]; 
		}
		a_curr[i] += bias_curr[i];
		z_curr[i] = computeFuncH(a_curr[i] , network->type[level]);
	}
	/*DEBUG
	printf("\nDEBUG:\n");
	for(int i = 0 ; i < size ; i ++)
		printf("%f \n" , z_curr[i]);
	DEBUG COMPLETE*/

	return;
}


void reverse_propagate_level(int level , BPN *network , double* target){
	double *weight_next , *delta_curr , *delta_next , *z_curr , *a_curr;
	int size , size_next;

	if(level == 0){//output level
		if(target == NULL)
			return;

		
		getLevelNodes(network , level , &z_curr , &a_curr , NULL , &delta_curr , &size);

		for(int i = 0 ; i < size ; i ++)
			delta_curr[i] = (z_curr[i] - target[i]) * computeDiffH(a_curr[i] , network->type[level]);

		return;
	}

	getLevelWeights(network , level - 1 , &weight_next , &size , &size_next);

	getLevelNodes(network , level , NULL , &a_curr , NULL , &delta_curr , &size);

	getLevelNodes(network , level - 1 , NULL , NULL , NULL , &delta_next , &size_next);

	for(int i = 0 ; i < size ; i ++){
		delta_curr[i] = 0; 
		for(int j = 0 ; j < size_next ; j ++)
			delta_curr[i] += delta_next[j] * *(weight_next + j * size + i);
		delta_curr[i] *= computeDiffH(a_curr[i] , network->type[level]);
	}

	return;
}

void weight_bias_update(BPN *network){
	double *weight = network->weight;
	double *delta = network->delta;
	double *z_prev = network->z_val + network->nodeSize[0];

	double *bias = network->bias;

	for(int i = 0 ; i < network->noLevels - 1; i ++){
					
		for(int j = 0 ; j < network->nodeSize[i] ; j ++){
			for(int k = 0 ; k < network->nodeSize[i + 1] ; k ++){
				*(weight + j * network->nodeSize[i + 1] + k) -= network->training_rate * *(delta + j) * *(z_prev + k);
			}

			*(bias + j) -= network->training_rate * *(delta + j);

		}

		weight += network->nodeSize[i] * network->nodeSize[i + 1];
		delta += network->nodeSize[i];
		z_prev += network->nodeSize[i + 1];
		bias += network->nodeSize[i];
	}

}

void forward(BPN* network , double* input){
	for(int i = network->noLevels - 1 ; i > -1 ; i --){
		if(i == network->noLevels - 1){
			forward_propagate_level(i , network , input);
			continue;
		}
		forward_propagate_level(i , network , NULL);
	}
}

double reverse(BPN* network , double* target){
	for(int i = 0 ; i < network->noLevels - 1 ; i ++){
		if(i == 0)
			reverse_propagate_level(i , network , target);
		else
			reverse_propagate_level(i , network , NULL);
	}

	double error = 0;
	for(int i = 0 ; i < network->nodeSize[0] ; i ++){
		error += (target[i] - network->z_val[i]) * (target[i] - network->z_val[i]);
	}

	return error;
}


int train(BPN* network , double* input , double* output , int dataset_no , int input_size , int output_size , int total_iterations){
	double error;
	double *ip , *op;
	int count = 0;

	if(total_iterations == -1)
		total_iterations = 1000;

	while(true){
		error = 0;
		ip = input;
		op = output;
		for(int i = 0 ; i < dataset_no ; i ++){
			forward(network , ip);
			error += reverse(network , op);
			weight_bias_update(network);
			ip = ip + input_size;
			op = op + output_size;
		}

//		printf("\nError:%f\n" , error);

		if(error < THRES || count == total_iterations)
			break;

		count ++;	
	}
	return count;
}

void initialize(BPN* network , int* noNodes , int levels , Type* type , double rate){

	network->noLevels = levels;
	network->nodeSize = new int[levels];
	network->type = new Type[levels];
	network->training_rate = rate;

	for(int i = 0 ; i < levels ; i ++){
		network->nodeSize[i] = noNodes[i];
		network->type[i] = type[i];
	}

	int numNodes = 0;
	int numWeights = 0;
	for(int i = 0 ; i < levels ; i ++){
		numNodes += noNodes[i];

		if(i == 0)
			continue;

		numWeights += noNodes[i] * noNodes[i - 1];
	}
	
	network->a_val = new double[numNodes];
	network->z_val = new double[numNodes];
	network->delta = new double[numNodes];
	network->bias = new double[numNodes];
	network->weight = new double[numWeights];

	network->noNodes = numNodes;
	network->noWeight = numWeights;

	
	time_t t;
	srand((unsigned)time(&t));

	for(int i = 0 ; i < numWeights ; i ++){
		
		double wt_in = (double)(rand() % 50) / 10000.0;
		wt_in = wt_in == 0.0 ? 0.0001 : wt_in;
		network->weight[i] = wt_in;

		if(i < numNodes){
			double nd_in = (double)(rand() % 50) / 10000.0;
			nd_in = nd_in == 0.0 ? 0.0001 : nd_in;
			network->a_val[i] = nd_in;
			network->z_val[i] = nd_in;
			network->bias[i] = nd_in;
			network->delta[i] = nd_in;
		}

	}

	if(numWeights == 2){//If number of weight connections is true, then no-weights = no-nodes + 1
		network->a_val[2] = 0.0001;
		network->z_val[2] = 0.0001;
		network->bias[2] = 0.0001;
		network->delta[2] = 0.0001;
	}

	
}

void returnOutput(BPN* network , double* input , double* output){
	int size = network->nodeSize[0];
	forward(network , input);

	for(int i = 0 ; i < size ; i ++){
		output[i] = network->z_val[i];
	}

	return;
}