#include "img_dataset_creator.h"


void createImageVector(char* filename , int size , double* out){
	cimg_library::CImg<double> image(filename);
	image.resize(size , size , 1 , 1 , 5);


	cimg_forXY(image , x , y){
	//	out[x * size + y] = image(x , y , 0 , 0) / 100000; for n = 20
		out[x * size + y] = image(x , y , 0 , 0) / 1000000;
	}
	return;
}