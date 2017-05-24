#include "common.h"
#include "bpn.h"
#include "bpn_cuda.h"

#include "img_dataset_creator.h"
#include "mnist.h"

#define TEST_OUTPUT "output.txt"

struct __data{
	double cpu_time;
	double gpu_time;
	double no_nodes;
	double no_weights;
	double no_layers;
	double no_iterations;
};

std::ofstream outfile_bin , outfile;

void write_file_binary(__data data){
	if(!outfile_bin.is_open())
		outfile.open("output.bin-utf8" , std::ios::binary);

	outfile.write((char*)&data , sizeof(data));
	return;
}

void close_file(){
	outfile.close();
	return;
}

void write_file(__data data){
	if(!outfile.is_open())
		outfile.open("output.txt");
	outfile<<"For Node Count: "<<data.no_nodes;
	outfile<<"\n\tTotal Number of Layers: "<<data.no_layers;
	outfile<<"\n\tTotal Number of Weights: "<<data.no_weights;
	outfile<<"\n\tTotal Iterations: "<<data.no_iterations;
	
	outfile<<"\n\tCPU Time: "<<data.cpu_time;
	outfile<<"\n\tGPU Time: "<<data.gpu_time<<"\n\n\n";
	return;
}


#define DEFAULT_TEST_CONFIG "test.conf"

void read_from_file(int** no_layers , int** no_nodes , int** no_minst , int** iterations , int& total){
	std::ifstream infile(DEFAULT_TEST_CONFIG);
	std::istream_iterator<int> begin(infile);
	std::istream_iterator<int> end;

	std::vector<int> contents;
	std::copy(begin , end , std::back_inserter(contents));

	total = contents.size() / 4;

	*no_layers = new int[total];
	*no_nodes = new int[total];
	*iterations = new int[total];
	*no_minst = new int[total];

	for(int i = 0 ; i < total ; i ++){
		*(*no_layers + i) = contents[i * 4 + 0];
		*(*no_nodes + i) = contents[i * 4 + 1];
		*(*no_minst + i) = contents[i * 4 + 2];
		*(*iterations + i) = contents[i * 4 + 3];
	}

	return;
}

Type* create_type(int node_count , Type t = Linear){
	Type* type = new Type[node_count];
	for(int i = 0 ; i < node_count - 1 ; i ++)
		type[i] = t;

	return type;
}

int* create_layers(int n_count , int input_size , int n_layers){
	int* layers = new int[n_layers];
	int lambda = 2 * n_count / n_layers;

	if(lambda <= input_size){
		printf("Fatal: Input vector too large for the given node count.\n");
		exit(1);
	}

	for(int i = 0 ; i < n_layers ; i ++){
		if(i % 4 == 0)
			layers[i] = input_size;
		else if(i % 4 == 1)
			layers[i] = lambda - 1;
		else if(i % 4 == 2)
			layers[i] = lambda - input_size;
		else
			layers[i] = 1;
	}

	return layers;
}

int main(){

	
	int *total_layers , *total_nodes , *iterations , *sample_size , total;
	
	read_from_file(&total_layers , &total_nodes , &sample_size , &iterations , total);

	int d_num , d_rows , d_cols , d_numl;
	double *dataset , *label;

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

	__data _write_out;
	
	double rate = 0.0001;

	CPU::BPN* network;
	network = new CPU::BPN;
	
	GPU::BPN_CUDA* _network;
	_network = new GPU::BPN_CUDA;

	//Begin Iteration
	for(int i = 0 ; i < total ; i ++){
		Type *type = create_type(total_layers[i]);
		int *layers = create_layers(total_nodes[i] , d_rows * d_cols , total_layers[i]);
		double time_spent;
	
		_write_out.no_layers = total_layers[i];
		_write_out.no_nodes = total_nodes[i];
		_write_out.no_iterations = iterations[i];

		for(int j = 0 ; j < 2 ; j ++){
			
			if(j == 0){
				CPU::initialize(network , layers , total_layers[i] , type , rate);
				clock_t begin = clock();
				int count = CPU::train(network , dataset , label , sample_size[i] , d_rows * d_cols , 1 , iterations[i]);
				clock_t end = clock();

				_write_out.no_weights = network->noWeight;
				_write_out.cpu_time = (double)(end - begin) / (CLOCKS_PER_SEC * count) * 1000;

			}
			else{
				GPU::initialize(_network , layers , total_layers[i] , type , rate);
				clock_t begin = clock();
				int count = GPU::train(_network , dataset , label , sample_size[i] , d_rows * d_cols , 1 , iterations[i]);
				clock_t end = clock();

				_write_out.gpu_time = (double)(end - begin) / (CLOCKS_PER_SEC * count) * 1000;
	
			}

		}
		printf("\n\nNetwork Information:\n");
		printf("\tNumber of Nodes : %d \n Number of Weight connections : %d \n" , _write_out.no_nodes , _write_out.no_weights);
		printf("\tCPU Time : %f\n\tGPU Time : %f" , _write_out.cpu_time , _write_out.gpu_time);
			
		write_file(_write_out);
		write_file_binary(_write_out);

		delete[] type , layers;

	}
	return 0;
}

