#include <chrono>
#include <climits>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <math.h>
#include <string.h>
#include <cstdlib>
#include <time.h>
#include <float.h>
#include <array>

using namespace std;

typedef unsigned char uchar;
typedef unsigned long ulong;

typedef struct
{
	double red, green, blue;
} RGB_Pixel;

typedef struct
{
	int width, height;
	int size;
	RGB_Pixel* data;
} RGB_Image;

typedef struct
{
	int size;
	RGB_Pixel center;
} RGB_Cluster;

/* Mersenne Twister related constants */
#define N 624
#define M 397
#define MAXBIT 30
#define MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define UPPER_MASK 0x80000000UL /* most significant w-r bits */
#define LOWER_MASK 0x7fffffffUL /* least significant r bits */
#define MAX_RGB_DIST 195075
#define NUM_RUNS 100

static ulong mt[N]; /* the array for the state vector  */
static int mti = N + 1; /* mti == N + 1 means mt[N] is not initialized */

/* initializes mt[N] with a seed */
void 
init_genrand(ulong s)
{
	mt[0] = s & 0xffffffffUL;
	for (mti = 1; mti < N; mti++)
	{
		mt[mti] =
			(1812433253UL * (mt[mti - 1] ^ (mt[mti - 1] >> 30)) + mti);
		/* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
		/* In the previous versions, MSBs of the seed affect   */
		/* only MSBs of the array mt[].                        */
		/* 2002/01/09 modified by Makoto Matsumoto             */
		mt[mti] &= 0xffffffffUL;
		/* for >32 bit machines */
	}
}

ulong 
genrand_int32(void)
{
	ulong y;
	static ulong mag01[2] = { 0x0UL, MATRIX_A };
	/* mag01[x] = x * MATRIX_A  for x = 0, 1 */

	if (mti >= N)
	{ /* generate N words at one time */
		int kk;

		if (mti == N + 1)
		{
			/* if init_genrand ( ) has not been called, */
			init_genrand(5489UL); /* a default initial seed is used */
		}

		for (kk = 0; kk < N - M; kk++)
		{
			y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
			mt[kk] = mt[kk + M] ^ (y >> 1) ^ mag01[y & 0x1UL];
		}

		for (; kk < N - 1; kk++)
		{
			y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
			mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
		}

		y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
		mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ mag01[y & 0x1UL];
		mti = 0;
	}

	y = mt[mti++];

	/* Tempering */
	y ^= (y >> 11);
	y ^= (y << 7) & 0x9d2c5680UL;
	y ^= (y << 15) & 0xefc60000UL;
	y ^= (y >> 18);

	return y;
}

double 
genrand_real2(void)
{
	return genrand_int32() * (1.0 / 4294967296.0);
	/* divided by 2^32 */
}

/* Function for generating a bounded random integer between 0 and RANGE */
/* Source: http://www.pcg-random.org/posts/bounded-rands.html */

uint32_t bounded_rand(const uint32_t range)
{
	uint32_t x = genrand_int32();
	uint64_t m = ((uint64_t)x) * ((uint64_t)range);
	uint32_t l = (uint32_t)m;

	if (l < range)
	{
		uint32_t t = -range;

		if (t >= range)
		{
			t -= range;
			if (t >= range)
			{
				t %= range;
			}
		}

		while (l < t)
		{
			x = genrand_int32();
			m = ((uint64_t)x) * ((uint64_t)range);
			l = (uint32_t)m;
		}
	}

	return m >> 32;
}

RGB_Image* 
read_PPM(const char* filename)
{
	uchar byte;
	char buff[16];
	int c, max_rgb_val, i = 0;
	FILE* fp;
	RGB_Pixel* pixel;
	RGB_Image* img;

	fp = fopen(filename, "rb");
	if (!fp)
	{
		fprintf(stderr, "Unable to open file '%s'!\n", filename);
		exit(EXIT_FAILURE);
	}

	/* read image format */
	if (!fgets(buff, sizeof(buff), fp))
	{
		perror(filename);
		exit(EXIT_FAILURE);
	}

	/*check the image format to make sure that it is binary */
	if (buff[0] != 'P' || buff[1] != '6')
	{
		fprintf(stderr, "Invalid image format (must be 'P6')!\n");
		exit(EXIT_FAILURE);
	}

	img = (RGB_Image*)malloc(sizeof(RGB_Image));
	if (!img)
	{
		fprintf(stderr, "Unable to allocate memory!\n");
		exit(EXIT_FAILURE);
	}

	/* skip comments */
	c = getc(fp);
	while (c == '#')
	{
		while (getc(fp) != '\n');
		c = getc(fp);
	}

	ungetc(c, fp);

	/* read image dimensions */
	if (fscanf(fp, "%u %u", &img->width, &img->height) != 2)
	{
		fprintf(stderr, "Invalid image dimensions ('%s')!\n", filename);
		exit(EXIT_FAILURE);
	}

	/* read maximum component value */
	if (fscanf(fp, "%d", &max_rgb_val) != 1)
	{
		fprintf(stderr, "Invalid maximum R, G, B value ('%s')!\n", filename);
		exit(EXIT_FAILURE);
	}

	/* validate maximum component value */
	if (max_rgb_val != 255)
	{
		fprintf(stderr, "'%s' is not a 24-bit image!\n", filename);
		exit(EXIT_FAILURE);
	}

	while (fgetc(fp) != '\n');

	/* allocate memory for pixel data */
	img->size = img->height * img->width;
	img->data = (RGB_Pixel*)malloc(img->size * sizeof(RGB_Pixel));

	if (!img)
	{
		fprintf(stderr, "Unable to allocate memory!\n");
		exit(EXIT_FAILURE);
	}

	/* Read in pixels using buffer */
	while (fread(&byte, 1, 1, fp) && i < img->size)
	{
		pixel = &img->data[i];
		pixel->red = byte;
		fread(&byte, 1, 1, fp);
		pixel->green = byte;
		fread(&byte, 1, 1, fp);
		pixel->blue = byte;
		i++;
	}

	fclose(fp);

	return img;
}

void 
write_PPM(const RGB_Image* img, const char* filename)
{
	uchar byte;
	FILE* fp;

	fp = fopen(filename, "wb");
	if (!fp)
	{
		fprintf(stderr, "Unable to open file '%s'!\n", filename);
		exit(EXIT_FAILURE);
	}

	fprintf(fp, "P6\n");
	fprintf(fp, "%d %d\n", img->width, img->height);
	fprintf(fp, "%d\n", 255);

	for (int i = 0; i < img->size; i++)
	{
		byte = (uchar)img->data[i].red;
		fwrite(&byte, sizeof(uchar), 1, fp);
		byte = (uchar)img->data[i].green;
		fwrite(&byte, sizeof(uchar), 1, fp);
		byte = (uchar)img->data[i].blue;
		fwrite(&byte, sizeof(uchar), 1, fp);
	}

	fclose(fp);
}

/* Function to generate random cluster centers. */
RGB_Cluster* 
gen_rand_centers(const RGB_Image* img, const int k) {
	printf("Generating centers...\n");
	RGB_Pixel rand_pixel;
	RGB_Cluster* cluster;

	cluster = (RGB_Cluster*)malloc(k * sizeof(RGB_Cluster));

	for (int i = 0; i < k; i++) {
		/* Make the initial guesses for the centers, m1, m2, ..., mk */
		rand_pixel = img->data[bounded_rand(img->size)];

		cluster[i].center.red = rand_pixel.red;
		cluster[i].center.green = rand_pixel.green;
		cluster[i].center.blue = rand_pixel.blue;

		/* Set the number of points assigned to k cluster to zero, n1, n2, ..., nk */
		cluster[i].size = 0;

		// cout << "\nCluster Centers: " << cluster[i].center.red << ", " << cluster[i].center.green <<", "<<  cluster[i].center.blue;
	}

	return(cluster);
}

/*
   For application of the batchk k-means algorithm to color quantization, see
   M. E. Celebi, Improving the Performance of K-Means for Color Quantization,
   Image and Vision Computing, vol. 29, no. 4, pp. 260-271, 2011.
 */
 /* Color quantization using the batch k-means algorithm */
void 
batch_kmeans(const RGB_Image* img, const int num_colors,
	const int max_iters, RGB_Cluster* clusters)
{
	int num_pixels = img->size; /*Get the number of pixels in the image*/
	int* assign = new int[num_pixels]; /*Array to store the assignment of each pixel to a cluster*/
	double sse; /*Variable to store SSE*/
	double delta_red, delta_green, delta_blue;
	double dist;
	RGB_Pixel pixel;
	RGB_Cluster *temp_clusters;


	temp_clusters = ( RGB_Cluster * ) malloc ( num_colors * sizeof ( RGB_Cluster ) );

	/*Loop until the max number of iterations is hit : Terminate by iterations only*/
	for (int iter = 0; iter < max_iters; iter++){
		
		sse = 0.0; /*Reset sse for next iteration*/

		/*Reset the clusters for the next iteration*/
		for (int j = 0; j < num_colors; j++){
			temp_clusters[j].center.red = 0.0;
			temp_clusters[j].center.green = 0.0;
			temp_clusters[j].center.blue = 0.0;
			temp_clusters[j].size = 0;
			clusters[j].size = 0; 
		}
		/*Loop over all pixels and assign them to the nearest cluster*/
		for(int i = 0; i < num_pixels; i++){
			double min_dist = MAX_RGB_DIST; /*store the max distance in a variable*/
			int cluster_index = 0;
			pixel = img->data[i];
			
			/*Calculate the Euclidean distance between the current pixel and current cluster*/
			for (int j = 0; j < num_colors; j++){
				delta_red = pixel.red - clusters[j].center.red;
				delta_green = pixel.green - clusters[j].center.green;
				delta_blue = pixel.blue - clusters[j].center.blue;
				dist = delta_red * delta_red + delta_green * delta_green + delta_blue * delta_blue;

				/*Checking if this is the closest center*/
				if (dist < min_dist){
					min_dist = dist; /*Resetting min_dist*/
					cluster_index = j; /*Assigning the closest pixel to a center*/
				}
			}
			assign[i] = cluster_index; /*Store the assignment of the pixel to cluster in the array*/
			clusters[cluster_index].size++; /*Increase size of cluster based on the assigned index*/
			sse += min_dist; /*Accumulate the sse based on the euclidean distance of all pixels*/

			/* Update the temporary center & size of the nearest cluster */
			temp_clusters[cluster_index].center.red += pixel.red;
			temp_clusters[cluster_index].center.green += pixel.green;
			temp_clusters[cluster_index].center.blue += pixel.blue;
		}

		/*Update cluster centers*/
		for (int j = 0; j < num_colors; j++){
			
			int cluster_size = clusters[j].size; /*Getting the size of each cluster*/

			/*Center update*/
			clusters[j].center.red = temp_clusters[j].center.red / cluster_size;
			clusters[j].center.green = temp_clusters[j].center.green / cluster_size;
			clusters[j].center.blue = temp_clusters[j].center.blue / cluster_size;
			
		}
		
		cout << "Iteration " << iter + 1 << ": " << "SSE = " << sse << endl;

	}
}

/*Accelerated version of batch_kmeans*/
void 
tie_algorithm(const RGB_Image* img, const int num_colors,
	const int max_iters, RGB_Cluster* clusters)
	{
		int num_pixels = img->size; /*Get the number of pixels in the image*/

		/*2d array to store distances between centers*/
		double** center_to_center_dist_original = new double*[num_colors];
		for(int i = 0; i < num_colors; i++)/*Allocate memory for each row*/
		{ 
			center_to_center_dist_original[i] = new double[num_colors];
		}

		// Create an index array to store indices
		int** center_to_center_dist_sorted = new int*[num_colors];
		for (int i = 0; i < num_colors; ++i) {
			center_to_center_dist_sorted[i] = new int[num_colors];
			for (int j = 0; j < num_colors; ++j) {
				center_to_center_dist_sorted[i][j] = j;
			}
		}

		int* assign = new int[num_pixels];
		int* pixel_nearest_center = new int[num_pixels];

		int nearest_center_index; 
		int temp_nearest_index;
		double sse; /*Variable to store SSE*/
		double delta_red, delta_green, delta_blue, delta_red_temp, delta_green_temp, delta_blue_temp;
		double nearest_center_distance, nearest_center_distance_temp;
		double dist;
		RGB_Pixel pixel;
		RGB_Cluster *temp_clusters;

		temp_clusters = ( RGB_Cluster * ) malloc ( num_colors * sizeof ( RGB_Cluster ) );

		/*Initialize pixel center array*/
		for(int i = 0; i < num_pixels; i++)
		{
			pixel_nearest_center[i] = 1;
		}


		for (int iter = 0; iter < max_iters; iter++)
		{

			sse = 0.0; /*Reset sse for next iteration*/

			/*Reset the clusters for the next iteration*/
			for (int j = 0; j < num_colors; j++)
			{
				temp_clusters[j].center.red = 0.0;
				temp_clusters[j].center.green = 0.0;
				temp_clusters[j].center.blue = 0.0;
				temp_clusters[j].size = 0;
				clusters[j].size = 0; 
			}

			/*Computer pairwise distances between each center*/
			for (int i = 0; i < num_colors; i++)
			{
				double min_dist = MAX_RGB_DIST; /*store the max distance in a variable*/

				for (int j = 0; j < num_colors; j++)
				{
					delta_red = clusters[i].center.red - clusters[j].center.red;
					delta_green = clusters[i].center.green - clusters[j].center.green;
					delta_blue = clusters[i].center.blue - clusters[j].center.blue;

					center_to_center_dist_original[i][j] = 0.25 * (delta_red * delta_red + delta_green * delta_green + delta_blue * delta_blue);
				}
				
				
			}


			/*Sort each row  separately in ascending order*/
			for(int i = 0; i < num_colors; i++)
			{
				sort(center_to_center_dist_sorted[i], center_to_center_dist_sorted[i] + num_colors, 
				[&](const int& a, const int& b) {
					return center_to_center_dist_original[i][a] < center_to_center_dist_original[i][b];
				});
			}


			/*Assign each pixel to its nearest center*/
			for(int i = 0; i < num_pixels; i++)
			{
				nearest_center_index = pixel_nearest_center[i]; /*Index of current nearest center*/
				pixel = img->data[i];

				delta_red = pixel.red - clusters[nearest_center_index].center.red;
				delta_green = pixel.green - clusters[nearest_center_index].center.green;
				delta_blue = pixel.blue - clusters[nearest_center_index].center.blue;

				nearest_center_distance = delta_red * delta_red + delta_green * delta_green + delta_blue * delta_blue;

				/*Update the pixels nearest center if necessary*/
				for (int j = 0+1; j < num_colors; j++)
				{
					if (nearest_center_distance < center_to_center_dist_original[nearest_center_index][j])
					{
						break;
					}

					/*Possibility that this is the current pixels nearest center*/
					temp_nearest_index = center_to_center_dist_sorted[nearest_center_index][j];

					delta_red_temp = pixel.red - clusters[temp_nearest_index].center.red;
					delta_green_temp = pixel.green - clusters[temp_nearest_index].center.green;
					delta_blue_temp = pixel.blue - clusters[temp_nearest_index].center.blue;

					nearest_center_distance_temp = delta_red * delta_red + delta_green * delta_green + delta_blue * delta_blue;
					
					/*The temp nearest center is closer to the pixel than its current nearest center*/
					if(nearest_center_distance_temp < nearest_center_distance || 
					((nearest_center_distance_temp == nearest_center_distance) && 
					(temp_nearest_index < nearest_center_index)))
					{	
						/*Update nearest center information*/
						nearest_center_distance = nearest_center_distance_temp; /*Curren nearest center distance*/
						nearest_center_index = temp_nearest_index; /*Current nearest center index*/
						j = 1; /*Reset search*/
					}
						

				}
				
				pixel_nearest_center[i] = nearest_center_index; /*Assign pixel to its new nearest center*/
				clusters[nearest_center_index].size++; /*Increase cluster size based on assigned index*/
				sse += nearest_center_distance;

				/* Update the temporary center & size of the nearest cluster */
				temp_clusters[nearest_center_index].center.red += pixel.red;
				temp_clusters[nearest_center_index].center.green += pixel.green;
				temp_clusters[nearest_center_index].center.blue += pixel.blue;
				
			}
			

			/*Update cluster centers*/
			for (int j = 0; j < num_colors; j++)
			{
				int cluster_size = clusters[j].size; /*Getting the size of each cluster*/

				/*Center update*/
				clusters[j].center.red = temp_clusters[j].center.red / cluster_size;
				clusters[j].center.green = temp_clusters[j].center.green / cluster_size;
				clusters[j].center.blue = temp_clusters[j].center.blue / cluster_size;

				
			}

			cout << "Iteration " << iter + 1 << " SSE: " << sse << endl;
		}
	}

/*Janceys Kmeans algorithm - Similar to batch kmeans, with a different center update step*/
void 
janceys_kmeans(const RGB_Image* img, const int num_colors,
	const int max_iters, RGB_Cluster* clusters)
	{
		int num_pixels = img->size; /*Get the number of pixels in the image*/
		int* assign = new int[num_pixels]; /*Array to store the assignment of each pixel to a cluster*/
		double sse; /*Variable to store SSE*/
		double delta_red, delta_green, delta_blue;
		double dist;
		RGB_Pixel pixel;
		RGB_Cluster *temp_clusters;
		double red_centroid, prev_red_centroid, green_centroid, prev_green_centroid, blue_centroid, prev_blue_centroid;
		int alpha = 1.8;


		temp_clusters = ( RGB_Cluster * ) malloc ( num_colors * sizeof ( RGB_Cluster ) );

		/*Loop until the max number of iterations is hit : Terminate by iterations only*/
		for (int iter = 0; iter < max_iters; iter++){
			
			sse = 0.0; /*Reset sse for next iteration*/

			/*Reset the clusters for the next iteration*/
			for (int j = 0; j < num_colors; j++){
				temp_clusters[j].center.red = 0.0;
				temp_clusters[j].center.green = 0.0;
				temp_clusters[j].center.blue = 0.0;
				temp_clusters[j].size = 0;
				clusters[j].size = 0; 
			}
			/*Loop over all pixels and assign them to the nearest cluster*/
			for(int i = 0; i < num_pixels; i++){
				double min_dist = MAX_RGB_DIST; /*store the max distance in a variable*/
				int cluster_index = 0;
				pixel = img->data[i];
				
				/*Calculate the Euclidean distance between the current pixel and current cluster*/
				for (int j = 0; j < num_colors; j++){
					delta_red = pixel.red - clusters[j].center.red;
					delta_green = pixel.green - clusters[j].center.green;
					delta_blue = pixel.blue - clusters[j].center.blue;
					dist = delta_red * delta_red + delta_green * delta_green + delta_blue * delta_blue;

					/*Checking if this is the closest center*/
					if (dist < min_dist){
						min_dist = dist; /*Resetting min_dist*/
						cluster_index = j; /*Assigning the closest pixel to a center*/
					}
				}
				assign[i] = cluster_index; /*Store the assignment of the pixel to cluster in the array*/
				clusters[cluster_index].size++; /*Increase cluster size to make room for new pixel*/
				sse += min_dist; /*Accumulate the sse based on the euclidean distance of all pixels*/

				/* Update the temporary center & size of the nearest cluster */
				temp_clusters[cluster_index].center.red += pixel.red;
				temp_clusters[cluster_index].center.green += pixel.green;
				temp_clusters[cluster_index].center.blue += pixel.blue;
			}

			/*Recompute cluster centers*/
			for (int j = 0; j < num_colors; j++){
				
				int cluster_size = clusters[j].size; /*Getting the size of each cluster*/

				/*Store centroid of each cluster update*/
				red_centroid = temp_clusters[j].center.red / cluster_size;
				green_centroid = temp_clusters[j].center.green / cluster_size;
				blue_centroid = temp_clusters[j].center.blue / cluster_size;
				prev_red_centroid = temp_clusters[j-1].center.red / cluster_size;
				prev_green_centroid = temp_clusters[j-1].center.green / cluster_size;
				prev_blue_centroid = temp_clusters[j-1].center.blue / cluster_size;

				/*Update centers based on janceys kmeans using alpha*/
				clusters[j].center.red = (alpha * red_centroid) + ((1 - alpha) * prev_red_centroid);
				clusters[j].center.green = (alpha * green_centroid) + ((1 - alpha) * prev_green_centroid);
				clusters[j].center.blue = (alpha * blue_centroid) + ((1 - alpha) * prev_blue_centroid);


				// /*Center update*/
				// clusters[j].center.red = temp_clusters[j].center.red / cluster_size;
				// clusters[j].center.green = temp_clusters[j].center.green / cluster_size;
				// clusters[j].center.blue = temp_clusters[j].center.blue / cluster_size;
				
			}
			
			cout << "Iteration " << iter + 1 << ": " << "SSE = " << sse << endl;

		}
	}

/*Maximin algorithm to initialize k-means*/
RGB_Cluster *
maximin(const RGB_Image* img, const int num_colors)
	{	
		int num_pixels = img->size;
		double* d = new double[num_pixels];
		RGB_Cluster *cluster;
		RGB_Cluster centroid;
		RGB_Pixel pixel;
		double red_sum = 0.0, green_sum = 0.0, blue_sum = 0.0;
		double delta_red, delta_green, delta_blue;
		double dist;
		double max_dist;
		int next_cluster;

		cluster = ( RGB_Cluster * ) malloc ( num_colors * sizeof ( RGB_Cluster ) );

		/*Select the first center arbitrarily*/
		for(int i = 0; i < num_pixels; i++){
			pixel = img->data[i];
			/*Sum the RGB components of the pixels*/
			red_sum += pixel.red;
			green_sum += pixel.green;
			blue_sum += pixel.blue;
		}

		centroid.center.red = red_sum / num_pixels;
		centroid.center.green = green_sum / num_pixels;
		centroid.center.blue = blue_sum / num_pixels;

		cluster[0] = centroid; /*Set the first center to the calculated centroid*/
		cluster[0].size = 0;

		/*Set distances to 'infinity'*/
		for(int i = 0; i < num_pixels; i++){
			d[i] = MAX_RGB_DIST;
		}

		/*Calculate the remaining centers*/
		for(int j = 0 + 1; j < num_colors; j++){ /*Start at one because we have already assigned the first center (cluster[0])*/
			max_dist = -MAX_RGB_DIST; /*store the max distance in a variable*/
			next_cluster = 0;
			
			/*Calculate the Euclidean distance between the current pixel and the previous cluster*/
			for (int i = 0; i < num_pixels; i++){
				pixel = img->data[i];

				delta_red = cluster[j-1].center.red - pixel.red;
				delta_green = cluster[j-1].center.green - pixel.green;
				delta_blue = cluster[j-1].center.blue - pixel.blue;
				dist = delta_red * delta_red + delta_green * delta_green + delta_blue * delta_blue;

				/*Checking if this is the closest center*/
				if (dist < d[i]){
					d[i] = dist; /*Updating the distance between the pixel and closest center*/
				}

				/*Getting the furthest pixel away from the current center to choose as the next center*/
				if (max_dist < d[i]){
					max_dist = d[i];
					next_cluster = i;
				}
			}
				/*Assign the furthest pixel as a center*/
				cluster[j].center = img->data[next_cluster];
				cluster[j].size = 0; /*Reset cluster size to choose next center*/
		}
				return cluster;
	}


void
free_img(const RGB_Image* img) {
	/* Free Image Data*/
	free(img->data);

	/* Free Image Pointer*/
	delete(img);
}

int 
main(int argc, char* argv[])
{
	char* filename;						/* Filename Pointer*/
	int k;								/* Number of clusters*/
	RGB_Image* img;
	RGB_Image* out_img;
	RGB_Cluster* cluster;
	
	if (argc == 3) {
		/* Image filename */
		filename = argv[1];

		/* k, number of clusters */
		k = atoi(argv[2]);

	}
	else if (argc > 3) {
		printf("Too many arguments supplied.\n");
		return 0;
	}
	else {
		printf("Two arguments expected: image filename and number of clusters.\n");
		return 0;
	}

	srand(time(NULL));

	/* Print Args*/
	printf("%s %d\n", filename, k );

	/* Read Image*/
	img = read_PPM(filename);

	/* Test Batch K-Means*/
	/* Start Timer*/
	/*Declare data types of start, stop, and elapsed*/
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	/* Initialize centers */
    //cluster = gen_rand_centers(img, k);

	/*Initialize centers using maximin*/
	cluster = maximin(img, k);

	/* Implement Batch K-means*/
	tie_algorithm(img, k, INT_MAX, cluster);

	/* Stop Timer*/
	std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();

	/* Execution Time*/
	std::chrono::milliseconds elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

	free(cluster);

	return 0;
}

