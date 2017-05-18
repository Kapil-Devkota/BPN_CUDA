#include "mnist.h"

int reverseInt(int i){
	unsigned char u1 , u2 , u3 , u4;
	u1 =		 i & 255;
	u2 =  (i >> 8) & 255;
	u3 = (i >> 16) & 255;
	u4 = (i >> 24) & 255;
	return ((int)u1 << 24) + ((int)u2 << 16) + ((int)u3 << 8) + (int)u4;
}


double** ReadMNISTIMAGE(char *filename , int& num_image , int& size_image_rows , int& size_image_cols )
{
//    arr.resize(NumberOfImages,vector<double>(DataOfAnImage));
    std::ifstream file (filename , std::ios::binary);
	double **dataset = NULL;
	
	if (file.is_open())
    {
        int magic_number = 0;

        file.read((char*)&magic_number , sizeof(int));
        magic_number = reverseInt(magic_number);

        file.read((char*)&num_image , sizeof(int));
        num_image = reverseInt(num_image);
        
		file.read((char*)&size_image_rows , sizeof(size_image_rows));
        size_image_rows = reverseInt(size_image_rows);
 
		file.read((char*)&size_image_cols , sizeof(size_image_cols));
        size_image_cols = reverseInt(size_image_cols);

		dataset = new double*[num_image];
		
		int size_image = size_image_rows * size_image_cols;
		for(int i = 0 ; i < num_image ; i ++)
			dataset[i] = new double[size_image];

		for(int i = 0 ; i < num_image ; i ++)        
            for(int r = 0 ; r < size_image_rows ; r ++)
                for(int c = 0 ; c < size_image_cols ; c ++)
                {
                    unsigned char temp = 0;
                    file.read((char*)&temp , sizeof(temp));
                    dataset[i][(size_image_rows * r) + c] = (double)temp;
                }
            
        

    }

	return dataset;
}
 
double* ReadMNISTLABEL(char* filename , int& size){
    
	double *dataset = NULL;
	int n_rows , n_cols , mgk , mgk_;
	
	std::ifstream file (filename , std::ios::binary);
	

	if(file.is_open()){
		
		file.read((char*)&mgk , sizeof(int));
		mgk = reverseInt(mgk);
		
		file.read((char*)&size , sizeof(int));
		size = reverseInt(size);

		dataset = new double[size];

		for(int i = 0 ; i < size ; i ++){
			unsigned char tmp;
			file.read((char *)&tmp , 1);
			dataset[i] = (double)tmp;
			if(i % 10 == 0)
				printf("\n");
			printf("%1.0f  " , dataset[i]);
		}

	}

	return dataset;
}