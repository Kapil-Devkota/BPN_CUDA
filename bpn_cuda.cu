#include "bpn_cuda.h"

#define THRES 0.0001 /*Threshold of training error*/

__device__ double computeFunc(double x , Type t){
	if(t == Linear)
		return x;
	
	if(t == Sigmoidal){
		double out = 1 + exp(-x);
		return 1 / out;
	}
}

__device__ double computeDiff(double x , Type t){
	if(t == Linear)
		return 1;
	if(t == Sigmoidal){
		double out = computeFunc(x , t);
		return out * (1 - out);
	}
}

/*CUDA function to feed z value at the input*/
__global__ void forward_propagate_input(double* z_curr
										, double* bias
										, int size
										){

	
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id >= size)
		return;

	z_curr[id] += bias[id];
	return;
}
										

/* CUDA function to propagate z and a values from the input level(level[size - 1]) to the output level(level[0])*/
__global__ void forward_propagate_level(double* a_curr				/*a values of the nodes of current level*/ 
										, double* z_curr			/*z values of the nodes of the current level*/
										, double* weight			/*weight connections between current(l) and previous(l + 1) level*/
										, double* z_prev			/*z values of the nodes of the previous(l + 1) level*/
										, double* bias_curr			/*bias value of the current level*/
										, int prev					/*size of previous level*/
										, int curr					/*size of current level*/
										, Type t					/*threshold function type of the current level*/
										){

	int id = threadIdx.x + blockDim.x * blockIdx.x;

	if(id >= curr)
		return;

	int i;

	a_curr[id] = 0;
	for(i = 0 ; i < prev ; i ++)
		a_curr[id] += weight[id * prev + i] * z_prev[i];

	a_curr[id] += bias_curr[id];
	z_curr[id] = computeFunc(a_curr[id] , t);

	return;

}



/*Function to initialize the delta-values at the output*/

__global__ void reverse_propagate_output(double *delta_curr					/*delta value of the current level*/ 
										 , double* z_curr					/*z value of the current level*/
										 , double* target					/*target value at the output*/
										 , double* a_curr					/*a values at the output*/
										 , int size							/*size of the output level*/
										 , Type t							/*threshold value at the output*/
										 ){
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if(id >= size)
		return;
	
	delta_curr[id] = (z_curr[id] - target[id]) * computeDiff(a_curr[id] , t);

	return;
}

/*Function to propagate the delta-values from one level to another*/
__global__ void reverse_propagate_level(double *delta_curr				
										, double *delta_next
										, double* weight_next
										, double* a_curr
										, Type t						/*Type of threshold function at the current level*/
										, int size_next					/*Number of nodes at the next level*/ 
										, int size						/*Number of nodes at the current level*/
										){

	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if(id >= size)
		return;

	int i;
	delta_curr[id] = 0;

	for(i = 0 ; i < size_next ; i ++)
		delta_curr[id] += delta_next[i] * weight_next[i * size + id];

	delta_curr[id] *= computeDiff(a_curr[id] , t);

	return;
}
/*Function that updates weight between two levels*/
__global__ void weight_update(double* weight						/*Weight between current and previous level*/
							  , double* delta_curr
							  , double* z_prev
							  , int size							/*Size of current level*/
							  , int size_prev
							  , double rate							/*Rate at which the weights are updated*/
							  ){
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if(id >= size * size_prev)
		return;

	int curr_node = id / size_prev;
	int prev_node = id % size_prev;

	weight[curr_node * size_prev + prev_node] -= rate * delta_curr[curr_node] * z_prev[prev_node];
	return;
}

/*Function that updates the bias of a level*/
__global__ void bias_update(double* delta_curr
							, double* bias                          /*Bias of the current level*/
							, int size
							, double rate){
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if(id >= size)
		return;

	bias[id] -= rate * delta_curr[id];

	return;
}


void copyBPNinput(BPN_CUDA* network , double *input){
	int inputLevel = network->noLevels - 1;
	cudaMemcpy(network->z_val + network->noNodes - network->nodeSize[inputLevel] , input , network->nodeSize[network->noLevels - 1] * sizeof(double) , cudaMemcpyHostToDevice);
	cudaMemcpy(network->a_val + network->noNodes - network->nodeSize[inputLevel], input , network->nodeSize[network->noLevels - 1] * sizeof(double) , cudaMemcpyHostToDevice);
	return;
}

/*Function to compute the forward propagation of values*/
void forward(BPN_CUDA* network , double *input){
	int i , sizePrev , sizeCurr;
	double* a_curr , double* z_curr , double* weight , double* z_prev , double* bias_curr;
	Type t = network->type[network->noLevels - 2];

	copyBPNinput(network , input);



	sizePrev = network->nodeSize[network->noLevels - 1];
	sizeCurr = network->nodeSize[network->noLevels - 2];
	a_curr = network->a_val + network->noNodes - sizePrev - sizeCurr;
	z_curr = network->z_val + network->noNodes - sizePrev - sizeCurr;
	weight = network->weight + network->noWeight - sizePrev * sizeCurr;
	z_prev = network->z_val + network->noNodes - sizePrev;
	bias_curr = network->bias + network->noNodes - sizePrev - sizeCurr;

	forward_propagate_input<<<(sizePrev / 1024 + 1) , (sizePrev > 1024 ? 1024 : sizePrev)>>>(z_prev , bias_curr + sizeCurr , sizePrev);

	for(i = network->noLevels - 2 ; i > -1 ; i --){
		forward_propagate_level<<<(sizeCurr / 1024 + 1) , (sizeCurr > 1024 ? 1024 : sizeCurr)>>>(a_curr , z_curr , weight , z_prev , bias_curr , sizePrev , sizeCurr , t);
		
		if(i == 0)
			break;

		
		if(i == 0)
			break;

		z_prev = z_curr;

		sizePrev = sizeCurr;
		sizeCurr = network->nodeSize[i - 1];

		a_curr = a_curr - sizeCurr;
		z_curr = z_curr - sizeCurr;
		bias_curr = bias_curr - sizeCurr;
		weight = weight - sizePrev * sizeCurr;

	}


}
/*Function to compute the reverse propagation of values*/
double reverse(BPN_CUDA* network , double* target){

	double *delta_curr , *z_curr , *a_curr , *delta_next , *weight_next , *z_curr_h , *target_d;
	delta_curr = network->delta;
	z_curr = network->z_val;
	a_curr = network->a_val;
	int size = network->nodeSize[0] , size_next;
	Type t = network->type[0];
	double error = 0;

	z_curr_h = new double[size];
	cudaMemcpy(z_curr_h , z_curr , size * sizeof(double) , cudaMemcpyDeviceToHost);

	cudaMalloc((void**)&target_d , size * sizeof(double));
	cudaMemcpy(target_d , target , size * sizeof(double) , cudaMemcpyHostToDevice);
	
	for(int i = 0 ; i < network->nodeSize[0] ; i ++)
		error += (target[i] - z_curr_h[i]) * (target[i] - z_curr_h[i]);

	reverse_propagate_output<<<(size / 1024 + 1) , (size > 1024 ? 1024 : size)>>>(delta_curr , z_curr , target_d , a_curr , size , t);

	delta_next = delta_curr;
	weight_next = network->weight;

	z_curr = NULL;
	delta_curr = delta_curr + size;
	a_curr = a_curr + size;
	t = network->type[1];
	size_next = size;
	size = network->nodeSize[1];


	for(int i = 1 ; i < network->noLevels ; i ++){
		reverse_propagate_level<<<(size / 1024 + 1) , (size > 1024 ? 1024 : size)>>>(delta_curr , delta_next , weight_next , a_curr , t , size_next , size);

		if(i == network->noLevels - 1)
			break;

		delta_next = delta_curr;
		weight_next = weight_next + size * size_next;

		a_curr = a_curr + size;
		delta_curr = delta_curr + size;

		size_next = size;
		size = network->nodeSize[i + 1];
		t = network->type[i + 1];
		
	}

	return error;

}


void weight_bias_update(BPN_CUDA* network , double rate){

	double* weight = network->weight;
	double* delta_curr = network->delta;

	int size = network->nodeSize[0];

	double* z_prev = network->z_val + size;
	double* bias = network->bias;

	int size_prev = network->nodeSize[1];
	for(int i = 1 ; i < network->noLevels ; i ++){
		weight_update<<<(size * size_prev / 1024 + 1) , (size * size_prev > 1024 ? 1024 : size * size_prev)>>>(weight , delta_curr , z_prev , size , size_prev , rate);
		bias_update<<<(size / 1024 + 1) , (size > 1024 ? 1024 : size)>>>(delta_curr , bias , size , rate);

		if(i == network->noLevels - 1)
			break;

		weight = weight + size * size_prev;
		delta_curr = delta_curr + size;
		z_prev = z_prev + size_prev;
		bias = bias + size;

		size = size_prev;
		size_prev = network->nodeSize[i + 1];
	}
}

int train(BPN_CUDA* network , double* input , double* output , int dataset_no , int input_size , int output_size){
	double error;
	double *ip , *op;
	int count = 0;
	while(true){
		error = 0;
		ip = input;
		op = output;
		for(int i = 0 ; i < dataset_no ; i ++){
			forward(network , ip);
			error += reverse(network , op);
			weight_bias_update(network , network->training_rate);
			ip = ip + input_size;
			op = op + output_size;
		}

		//printf("%f\n" , error);

		if(error < THRES || count == 1000)
			break;

		count ++;
	}
	return count;
}

void initialize(BPN_CUDA* network , int* noNodes , int levels , Type* type , double rate){
	
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
	
	double *device_mem;
	
	cudaMalloc((void**)&network->a_val , numNodes * sizeof(double));
	cudaMalloc((void**)&network->z_val , numNodes * sizeof(double));
	cudaMalloc((void**)&network->delta , numNodes * sizeof(double));
	cudaMalloc((void**)&network->bias , numNodes * sizeof(double));
	cudaMalloc((void**)&network->weight , numWeights * sizeof(double));

	network->noNodes = numNodes;
	network->noWeight = numWeights;

	double* initval = new double[numNodes];
	double* initweight = new double[numWeights];

	time_t t;
	srand((unsigned)time(&t));

	for(int i = 0 ; i < numWeights ; i ++){
		
		initweight[i] = (double)(rand() % 50) / 100000.0;
		initweight[i] = initweight[i] == 0.0 ? 0.0001 : initweight[i];

		if(i < numNodes){
			initval[i] = (double)(rand() % 50) / 100000.0;
			initval[i] = initval[i] == 0.0 ? 0.0001 : initval[i];
		}

	}

	if(numWeights == 2)//If number of weight connections is true, then no-weights = no-nodes + 1
		initval[2] = 0.0001;

	cudaMemcpy(network->a_val , initval , numNodes * sizeof(double) , cudaMemcpyHostToDevice);	
	cudaMemcpy(network->z_val , initval , numNodes * sizeof(double) , cudaMemcpyHostToDevice);	
	cudaMemcpy(network->delta , initval , numNodes * sizeof(double) , cudaMemcpyHostToDevice);	
	cudaMemcpy(network->bias , initval , numNodes * sizeof(double) , cudaMemcpyHostToDevice);	
	cudaMemcpy(network->weight , initweight , numWeights * sizeof(double) , cudaMemcpyHostToDevice);

}

void returnOutput(BPN_CUDA* network , double* input , double* output){
	int size = network->nodeSize[0];
	forward(network , input);

	cudaMemcpy(output , network->z_val , size * sizeof(double) , cudaMemcpyDeviceToHost);

	return;
}