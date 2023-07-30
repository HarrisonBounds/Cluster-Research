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
#include <vector>

using namespace std;

typedef unsigned char uchar;
typedef unsigned long ulong;
typedef unsigned int uint;

typedef struct
{
	double red, green, blue, weight;
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
	RGB_Pixel* data;
} RGB_Table;

typedef struct
{
	int size;
	RGB_Pixel center;
} RGB_Cluster;

typedef struct Bucket_Entry* Bucket;
struct Bucket_Entry
{
	uint red;
	uint green;
	uint blue;
	uint count;
	Bucket next;
};
typedef Bucket* Hash_Table;

// definitions for the color hash table
#define HASH_SIZE 20023

#define HASH(R, G, B) ( ( ( ( long ) ( R ) * 33023 + \
                            ( long ) ( G ) * 30013 + \
                            ( long ) ( B ) * 27011 ) \
                          & 0x7fffffff ) % HASH_SIZE )

// definitions for the color hash table
#define HASH_SIZE 20023

/* Mersenne Twister related constants */
#define N 624
#define M 397
#define MAXBIT 30
#define MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define UPPER_MASK 0x80000000UL /* most significant w-r bits */
#define LOWER_MASK 0x7fffffffUL /* least significant r bits */
#define MAX_RGB_DIST 195075
#define NUM_RUNS 100
#define NEAR_ZERO 1e-6

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

/* Reset a given set of clusters */

RGB_Cluster*
reset_centers(RGB_Cluster* clusters, const int numColors)
{
	for (int i = 0; i < numColors; i++)
	{
		clusters[i].center.red = 0.0;
		clusters[i].center.green = 0.0;
		clusters[i].center.blue = 0.0;
		clusters[i].center.weight = 0.0;
		clusters[i].size = 0.0;
	}

	return clusters;
}

/* Duplicate a given set of clusters */

RGB_Cluster*
duplicate_centers(const RGB_Cluster* orig, const int numColors)
{
	RGB_Cluster* copy = (RGB_Cluster*)malloc(numColors * sizeof(RGB_Cluster));

	for (int i = 0; i < numColors; i++)
	{
		copy[i].center.red = orig[i].center.red;
		copy[i].center.green = orig[i].center.green;
		copy[i].center.blue = orig[i].center.blue;
		/*
		copy[i].center.weight = orig[i].center.weight;
		*/
		copy[i].size = orig[i].size;
	}

	return copy;
}

double
calc_MSE(const RGB_Image* img, const RGB_Cluster* clusters, const int numColors)
{
	double deltaR, deltaG, deltaB, dist, minDist, sse = 0.0;
	RGB_Pixel pixel;

	for (int i = 0; i < img->size; i++)
	{
		pixel = img->data[i];
		minDist = MAX_RGB_DIST;

		for (int j = 0; j < numColors; j++)
		{
			deltaR = clusters[j].center.red - pixel.red;
			deltaG = clusters[j].center.green - pixel.green;
			deltaB = clusters[j].center.blue - pixel.blue;

			dist = deltaR * deltaR + deltaG * deltaG + deltaB * deltaB;

			if (dist < minDist)
			{
				minDist = dist;
			}
		}

		sse += minDist;
	}

	return sse / img->size;
}

void
free_image(const RGB_Image* img)
{
	free(img->data);
	delete (img);
}

RGB_Image*
map_pixels(const RGB_Image* inImg, const RGB_Cluster* cluster, const int numColors)
{
	double deltaR, deltaG, deltaB, dist, minDist;
	int minIndex;
	RGB_Image* outImg;
	RGB_Pixel pixel;

	outImg = new RGB_Image;
	outImg->data = new RGB_Pixel[inImg->size];
	outImg->height = inImg->height;
	outImg->width = inImg->width;
	outImg->size = outImg->height * outImg->width;

	for (int i = 0; i < inImg->size; i++)
	{
		pixel = inImg->data[i];
		minDist = MAX_RGB_DIST;

		for (int j = 0; j < numColors; j++)
		{
			deltaR = cluster[j].center.red - pixel.red;
			deltaG = cluster[j].center.green - pixel.green;
			deltaB = cluster[j].center.blue - pixel.blue;

			dist = deltaR * deltaR + deltaG * deltaG + deltaB * deltaB;
			if (dist < minDist)
			{
				minDist = dist;
				minIndex = j;
			}
		}

		outImg->data[i].red = cluster[minIndex].center.red;
		outImg->data[i].green = cluster[minIndex].center.green;
		outImg->data[i].blue = cluster[minIndex].center.blue;
	}

	return outImg;
}


/* Sorts an array in ascending order */
vector<pair<double, int>>
sort_array(double arr[], const int n)
{
	/* Vector to store element with respective present index */
	vector<pair<double, int>> vp;

	/* Insert element in pair vector to keep track of previous indexes */
	for (int i = 0; i < n; ++i)
	{
		vp.push_back(make_pair(arr[i], i));
	}

	/* Sort pair vector */
	sort(vp.begin(), vp.end());

	/* Display sorted element with previous indexes corresponding to each element */
	/*
	cout << "\nElement\t"
		<< "index" << endl;
	for (int i = 0; i < vp.size(); i++) {
		cout << vp[i].first << "\t"
			<< vp[i].second << endl;
	}
	*/

	return vp;
}

RGB_Table*
calc_color_table(const RGB_Image* img)
{
	uint red, green, blue;
	int ih, index;
	double factor = 1. / img->size;
	RGB_Pixel pixel;
	Bucket bucket, tempBucket;

	Hash_Table hashTable = (Hash_Table)malloc(HASH_SIZE * sizeof(Bucket));
	RGB_Table* colorTable = (RGB_Table*)malloc(sizeof(RGB_Table));
	colorTable->size = 0;

	for (ih = 0; ih < HASH_SIZE; ih++)
	{
		hashTable[ih] = NULL;
	}

	/* Read in pixels using buffer */
	for (int i = 0; i < img->size; i++)
	{
		pixel = img->data[i];

		/* Add the pixels to the hash table */
		red = pixel.red;
		green = pixel.green;
		blue = pixel.blue;

		/* Determine the bucket */
		ih = HASH(red, green, blue);

		/* Search for the color in the bucket chain */
		for (bucket = hashTable[ih]; bucket != NULL; bucket = bucket->next)
		{
			if (bucket->red == red && bucket->green == green && bucket->blue == blue)
			{
				/* This color exists in the hash table */
				break;
			}
		}

		if (bucket != NULL)
		{
			/* This color exists in the hash table */
			bucket->count++;
		}
		else
		{
			colorTable->size++;

			/* Create a new bucket entry for this color */
			bucket = (Bucket)malloc(sizeof(struct Bucket_Entry));

			bucket->red = red;
			bucket->green = green;
			bucket->blue = blue;
			bucket->count = 1;
			bucket->next = hashTable[ih];
			hashTable[ih] = bucket;
		}
	}

	colorTable->data = (RGB_Pixel*)malloc(colorTable->size * sizeof(RGB_Pixel));

	index = 0;
	for (ih = 0; ih < HASH_SIZE; ih++)
	{
		for (bucket = hashTable[ih]; bucket != NULL; )
		{
			colorTable->data[index].red = bucket->red;
			colorTable->data[index].green = bucket->green;
			colorTable->data[index].blue = bucket->blue;
			colorTable->data[index].weight = bucket->count * factor;
			index++;

			/* Save the current bucket pointer */
			tempBucket = bucket;

			/* Advance to the next bucket */
			bucket = bucket->next;

			/* Free the current bucket */
			free(tempBucket);
		}
	}

	/* cout <<  "index = " << index << "; colorTable->size = " << colorTable->size << end; */

	free(hashTable);

	return colorTable;
}

void
free_table(const RGB_Table* table)
{
	free(table->data);
	delete (table);
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

/* Jancey Algorithm */
void
jancey(const RGB_Image* img, const int numColors, RGB_Cluster* clusters, int& numIters, const double alpha, const bool isBatch, double& mse)
{
	numIters = 0; /* initialize output variable */
	mse = 0.0;
	int numChanges, minIndex, size;
	double deltaR, deltaG, deltaB, dist, minDist;
	double sse;

	int* member = new int[img->size];

	RGB_Pixel pixel;
	RGB_Cluster* temp = new RGB_Cluster[numColors];

	do
	{
		numChanges = 0;
		sse = 0.0;

		reset_centers(temp, numColors);

		for (int i = 0; i < img->size; i++)
		{
			pixel = img->data[i];

			minDist = MAX_RGB_DIST;
			minIndex = 0;

			for (int j = 0; j < numColors; j++)
			{
				deltaR = clusters[j].center.red - pixel.red;
				deltaG = clusters[j].center.green - pixel.green;
				deltaB = clusters[j].center.blue - pixel.blue;

				dist = deltaR * deltaR + deltaG * deltaG + deltaB * deltaB;
				if (dist < minDist)
				{
					minDist = dist;
					minIndex = j;
				}
			}

			temp[minIndex].center.red += pixel.red;
			temp[minIndex].center.green += pixel.green;
			temp[minIndex].center.blue += pixel.blue;
			temp[minIndex].size += 1;

			if (minIndex != member[i])
			{
				numChanges += 1;
				member[i] = minIndex;
			}

			sse += minDist;
		}
		/*Update centers via batch k-means algorithm*/
		if (isBatch)
		{
			for (int j = 0; j < numColors; j++)
			{
				size = temp[j].size;

				if (size != 0)
				{
					clusters[j].center.red = temp[j].center.red / size;
					clusters[j].center.green = temp[j].center.green / size;
					clusters[j].center.blue = temp[j].center.blue / size;
				}
			}
		}
		/*Update centers via jancey algorithm*/
		else
		{
			for (int j = 0; j < numColors; j++)
			{
				size = temp[j].size;

				if (size != 0)
				{
					clusters[j].center.red += alpha * (temp[j].center.red / size - clusters[j].center.red);
					clusters[j].center.green += alpha * (temp[j].center.green / size - clusters[j].center.green);
					clusters[j].center.blue += alpha * (temp[j].center.blue / size - clusters[j].center.blue);
				}
			}
		}

		numIters += 1;

		//cout << "Iteration " << numIters << ": SSE = " << sse << " [" << "# changes = " << numChanges << "]" << endl;

	} while (numChanges != 0);

	mse = sse / img->size;

	delete[] member;
	delete[] temp;
}

/* Weighted Jancey algorithm */
void
weighted_jancey(const RGB_Table* colorTable, const int numColors, RGB_Cluster* clusters, int& numIters, const double alpha, const bool isBatch, double& mse)
{
	numIters = 0; /* initialize output variable */
	mse = 0.0;
	int numChanges, minIndex, size;
	double deltaR, deltaG, deltaB, dist, minDist;
	int colorTableSize = colorTable->size;
	double weight;
	int* member = new int[colorTable->size];

	RGB_Pixel pixel;
	RGB_Cluster* temp = new RGB_Cluster[numColors];

	do
	{
		numChanges = 0;
		mse = 0.0;

		reset_centers(temp, numColors);

		for (int i = 0; i < colorTableSize; i++)
		{
			pixel = colorTable->data[i];

			minDist = MAX_RGB_DIST;
			minIndex = 0;

			for (int j = 0; j < numColors; j++)
			{
				deltaR = clusters[j].center.red - pixel.red;
				deltaG = clusters[j].center.green - pixel.green;
				deltaB = clusters[j].center.blue - pixel.blue;

				dist = deltaR * deltaR + deltaG * deltaG + deltaB * deltaB;
				if (dist < minDist)
				{
					minDist = dist;
					minIndex = j;
				}
			}

			temp[minIndex].center.red += (pixel.weight * pixel.red);
			temp[minIndex].center.green += (pixel.weight * pixel.green);
			temp[minIndex].center.blue += (pixel.weight * pixel.blue);
			temp[minIndex].center.weight += pixel.weight;

			if (minIndex != member[i])
			{
				numChanges += 1;
				member[i] = minIndex;
			}

			mse += (pixel.weight * minDist);
		}

		/*Update centers via batch k-means algorithm*/
		if (isBatch)
		{
			for (int j = 0; j < numColors; j++)
			{
				weight = temp[j].center.weight;
				if (weight != 0)
				{
					clusters[j].center.red = temp[j].center.red / weight;
					clusters[j].center.green = temp[j].center.green / weight;
					clusters[j].center.blue = temp[j].center.blue / weight;
				}
			}
		}
		/*Update centers via jancey algorithm*/
		else
		{
			for (int j = 0; j < numColors; j++)
			{
				weight = temp[j].center.weight;
				if (weight != 0)
				{
					clusters[j].center.red += alpha * (temp[j].center.red / weight - clusters[j].center.red);
					clusters[j].center.green += alpha * (temp[j].center.green / weight - clusters[j].center.green);
					clusters[j].center.blue += alpha * (temp[j].center.blue / weight - clusters[j].center.blue);
				}
			}
		}

		numIters += 1;

		//cout << "Iteration " << numIters << ": MSE = " << mse << " [" << "# changes = " << numChanges << "]" << endl;

	} while (numChanges != 0);

	delete[] member;
	delete[] temp;
}

/* Jancey accelerated using TIE (Triangle Equality Elimination) */
void
TIE_jancey(const RGB_Image* img, const int numColors, RGB_Cluster* clusters, int& numIters, const double alpha, const bool isBatch, double& mse)
{
	numIters = 0;
	mse = 0.0;
	int numChanges, minIndex, size, tempIndex, oldMem;
	double deltaR, deltaG, deltaB, dist, minDist, delta;
	double sse;

	RGB_Cluster clustI, clustJ;
	RGB_Pixel pixel;
	RGB_Cluster* temp = new RGB_Cluster[numColors];

	int* member = (int*)calloc(img->size, sizeof(int));

	int** p = new int* [numColors]; /*center to center distance array sorted in ascending order by index*/
	double** ccDist = new double* [numColors]; /*2d array to store center to center distances*/
	for (int i = 0; i < numColors; i++)
	{
		p[i] = new int[numColors];
		ccDist[i] = new double[numColors];
	}

	vector<pair<double, int>> vector;

	do
	{
		numChanges = 0;
		sse = 0.0;

		reset_centers(temp, numColors);

		for (int i = 0; i < numColors; i++)
		{
			ccDist[i][i] = 0.0;
			clustI = clusters[i];

			for (int j = i + 1; j < numColors; j++)
			{
				clustJ = clusters[j];

				deltaR = clustI.center.red - clustJ.center.red;
				deltaG = clustI.center.green - clustJ.center.green;
				deltaB = clustI.center.blue - clustJ.center.blue;

				ccDist[i][j] = ccDist[j][i] = 0.25 * (deltaR * deltaR + deltaG * deltaG + deltaB * deltaB);
			}

			vector = sort_array(ccDist[i], numColors);

			for (int j = 0; j < numColors; j++)
			{
				ccDist[i][j] = vector[j].first;
				p[i][j] = vector[j].second;
			}
		}

		for (int i = 0; i < img->size; i++)
		{
			pixel = img->data[i];
			minIndex = oldMem = member[i];

			/* calculate the distance from the pixel to its old center */
			deltaR = clusters[minIndex].center.red - pixel.red;
			deltaG = clusters[minIndex].center.green - pixel.green;
			deltaB = clusters[minIndex].center.blue - pixel.blue;
			minDist = deltaR * deltaR + deltaG * deltaG + deltaB * deltaB;

			for (int j = 1; j < numColors; j++)
			{
				if (minDist < ccDist[minIndex][j])
				{
					break;
				}

				tempIndex = p[minIndex][j];

				deltaR = clusters[tempIndex].center.red - pixel.red;
				deltaG = clusters[tempIndex].center.green - pixel.green;
				deltaB = clusters[tempIndex].center.blue - pixel.blue;

				dist = deltaR * deltaR + deltaG * deltaG + deltaB * deltaB;
				delta = dist - minDist;

				if (delta < 0.0)
				{
					minDist = dist;
					minIndex = tempIndex;
					j = 0;	/* reset the search */
				}
				else if ((tempIndex < minIndex) && (fabs(delta) < NEAR_ZERO))
				{
					minIndex = tempIndex;
					j = 0;	/* reset the search */
				}

			}

			temp[minIndex].center.red += pixel.red;
			temp[minIndex].center.green += pixel.green;
			temp[minIndex].center.blue += pixel.blue;
			temp[minIndex].size += 1;

			if (minIndex != oldMem)
			{
				numChanges += 1;
				member[i] = minIndex;
			}

			sse += minDist;
		}

		if (isBatch)
		{
			for (int j = 0; j < numColors; j++)
			{
				size = temp[j].size;

				if (size != 0)
				{
					clusters[j].center.red = temp[j].center.red / size;
					clusters[j].center.green = temp[j].center.green / size;
					clusters[j].center.blue = temp[j].center.blue / size;
				}
			}
		}
		else
		{
			for (int j = 0; j < numColors; j++)
			{
				size = temp[j].size;

				if (size != 0)
				{
					clusters[j].center.red += alpha * (temp[j].center.red / size - clusters[j].center.red);
					clusters[j].center.green += alpha * (temp[j].center.green / size - clusters[j].center.green);
					clusters[j].center.blue += alpha * (temp[j].center.blue / size - clusters[j].center.blue);
				}
			}
		}

		numIters += 1;

		//cout << "Iteration " << numIters << ": SSE = " << sse << " [" << "# changes = " << numChanges << "]" << endl;

	} while (numChanges != 0);

	mse = sse / img->size;

	delete[] member;
	delete[] temp;
	for (int i = 0; i < numColors; i++)
	{
		delete[] p[i];
		delete[] ccDist[i];
	}
	delete[] p;
	delete[] ccDist;
}

void
weighted_TIE_jancey(const RGB_Table* colorTable, const int numColors, RGB_Cluster* clusters, int& numIters, const double alpha, const bool isBatch, double& mse)
{
	numIters = 0;
	mse = 0.0;
	int numChanges, minIndex, size, tempIndex, oldMem;
	int colorTableSize = colorTable->size;
	double deltaR, deltaG, deltaB, dist, minDist, delta;
	double weight;

	RGB_Cluster clustI, clustJ;
	RGB_Pixel pixel;
	RGB_Cluster* temp = new RGB_Cluster[numColors];

	int* member = (int*)calloc(colorTableSize, sizeof(int));

	int** p = new int* [numColors]; /*center to center distance array sorted in ascending order by index*/
	double** ccDist = new double* [numColors]; /*2d array to store center to center distances*/
	for (int i = 0; i < numColors; i++)
	{
		p[i] = new int[numColors];
		ccDist[i] = new double[numColors];
	}

	vector<pair<double, int>> vector;

	do
	{
		numChanges = 0;
		mse = 0.0;

		reset_centers(temp, numColors);

		for (int i = 0; i < numColors; i++)
		{
			ccDist[i][i] = 0.0;
			clustI = clusters[i];

			for (int j = i + 1; j < numColors; j++)
			{
				clustJ = clusters[j];

				deltaR = clustI.center.red - clustJ.center.red;
				deltaG = clustI.center.green - clustJ.center.green;
				deltaB = clustI.center.blue - clustJ.center.blue;

				ccDist[i][j] = ccDist[j][i] = 0.25 * (deltaR * deltaR + deltaG * deltaG + deltaB * deltaB);
			}

			vector = sort_array(ccDist[i], numColors);

			for (int j = 0; j < numColors; j++)
			{
				ccDist[i][j] = vector[j].first;
				p[i][j] = vector[j].second;
			}
		}

		for (int i = 0; i < colorTableSize; i++)
		{
			pixel = colorTable->data[i];
			minIndex = oldMem = member[i];

			/* calculate the distance from the pixel to its old center */
			deltaR = clusters[minIndex].center.red - pixel.red;
			deltaG = clusters[minIndex].center.green - pixel.green;
			deltaB = clusters[minIndex].center.blue - pixel.blue;
			minDist = deltaR * deltaR + deltaG * deltaG + deltaB * deltaB;

			for (int j = 1; j < numColors; j++)
			{
				if (minDist < ccDist[minIndex][j])
				{
					break;
				}

				tempIndex = p[minIndex][j];

				deltaR = clusters[tempIndex].center.red - pixel.red;
				deltaG = clusters[tempIndex].center.green - pixel.green;
				deltaB = clusters[tempIndex].center.blue - pixel.blue;

				dist = deltaR * deltaR + deltaG * deltaG + deltaB * deltaB;
				delta = dist - minDist;

				if (delta < 0.0)
				{
					minDist = dist;
					minIndex = tempIndex;
					j = 0;	/* reset the search */
				}
				else if ((tempIndex < minIndex) && (fabs(delta) < NEAR_ZERO))
				{
					minIndex = tempIndex;
					j = 0;	/* reset the search */
				}

			}

			temp[minIndex].center.red += (pixel.weight * pixel.red);
			temp[minIndex].center.green += (pixel.weight * pixel.green);
			temp[minIndex].center.blue += (pixel.weight * pixel.blue);
			temp[minIndex].center.weight += pixel.weight;

			if (minIndex != oldMem)
			{
				numChanges += 1;
				member[i] = minIndex;
			}

			mse += (pixel.weight * minDist);
		}

		if (isBatch)
		{
			for (int j = 0; j < numColors; j++)
			{
				weight = temp[j].center.weight;
				if (weight != 0)
				{
					clusters[j].center.red = temp[j].center.red / weight;
					clusters[j].center.green = temp[j].center.green / weight;
					clusters[j].center.blue = temp[j].center.blue / weight;
				}
			}
		}
		else
		{
			for (int j = 0; j < numColors; j++)
			{
				weight = temp[j].center.weight;

				if (weight != 0)
				{
					clusters[j].center.red += alpha * (temp[j].center.red / weight - clusters[j].center.red);
					clusters[j].center.green += alpha * (temp[j].center.green / weight - clusters[j].center.green);
					clusters[j].center.blue += alpha * (temp[j].center.blue / weight - clusters[j].center.blue);
				}
			}
		}

		numIters += 1;

		//cout << "Iteration " << numIters << ": MSE = " << mse << " [" << "# changes = " << numChanges << "]" << endl;

	} while (numChanges != 0);

	delete[] member;
	delete[] temp;
	for (int i = 0; i < numColors; i++)
	{
		delete[] p[i];
		delete[] ccDist[i];
	}
	delete[] p;
	delete[] ccDist;
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
	const int numColors = 4;
	const double alpha = 1.7;
	char* filename;						/* Filename Pointer*/
	int k[numColors] = {32, 64, 128, 256}; /* Number of clusters*/
	int numIters;
	double mse;							
	RGB_Image* img;
	RGB_Image* out_img;
	RGB_Cluster* cluster;
	RGB_Table* table; 
	
	// if (argc == 3) {
	// 	/* Image filename */
	// 	filename = argv[1];

	// 	/* k, number of clusters */
	// 	k = atoi(argv[2]);

	// }
	// else if (argc > 3) {
	// 	printf("Too many arguments supplied.\n");
	// 	return 0;
	// }
	// else {
	// 	printf("Two arguments expected: image filename and number of clusters.\n");
	// 	return 0;
	// }

	srand(time(NULL));

	/* Print Args*/
	//printf("%s %d\n", filename, k);

	//Image 1
	//#####################################################################

	/* Read Image*/
	filename = "image/4.2.03.ppm";
	img = read_PPM(filename);

	cout << "Image 1, 4.2.03.ppm" << endl;
	/*For each K value, perform the 8 algorithms in succession on different images*/
	for(int i = 0; i < numColors; i++)
	{
		cout << "K = " << k[i] << endl;
		cout << "===================" << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/*Initialize RGB table for weighted algorithms*/
		table = calc_color_table(img);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		jancey(img, k[i], cluster, numIters, alpha, true, mse); /*Batch k-means*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

		cout << filename << ", " << "batch k-means, " << mse << ", " << numIters << ", " << elapsed.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start1 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		jancey(img, k[i], cluster, numIters, alpha, false, mse); /*Jancey k-means*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop1 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop1 - start1);

		cout << filename << ", " << "jancey k-means, " << mse << ", " << numIters << ", " << elapsed1.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start2 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		weighted_jancey(table, k[i], cluster, numIters, alpha, true, mse); /*Weighted Batch-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop2 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed2 = std::chrono::duration_cast<std::chrono::milliseconds>(stop2 - start2);

		cout << filename << ", " << "weighted batch k-means, " << mse << ", " << numIters << ", " << elapsed2.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start3 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		weighted_jancey(table, k[i], cluster, numIters, alpha, false, mse); /*Weighted Jancey-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop3 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed3 = std::chrono::duration_cast<std::chrono::milliseconds>(stop3 - start3);

		cout << filename << ", " << "weighted jancey k-means, " << mse << ", " << numIters << ", " << elapsed3.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start4 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		TIE_jancey(img, k[i], cluster, numIters, alpha, true, mse); /*TIE + Batch-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop4 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed4 = std::chrono::duration_cast<std::chrono::milliseconds>(stop4 - start4);

		cout << filename << ", " << "TIE + batch k-means, " << mse << ", " << numIters << ", " << elapsed4.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start5 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		TIE_jancey(img, k[i], cluster, numIters, alpha, false, mse); /*TIE + Jancey-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop5 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed5 = std::chrono::duration_cast<std::chrono::milliseconds>(stop5 - start5);

		cout << filename << ", " << "TIE + jancey k-means, " << mse << ", " << numIters << ", " << elapsed5.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start6 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		weighted_TIE_jancey(table, k[i], cluster, numIters, alpha, true, mse); /*Weighted TIE + Batch-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop6 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed6 = std::chrono::duration_cast<std::chrono::milliseconds>(stop6 - start6);

		cout << filename << ", " << "weighted TIE + batch k-means, " << mse << ", " << numIters << ", " << elapsed6.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start7 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		weighted_TIE_jancey(table, k[i], cluster, numIters, alpha, false, mse); /*Weighted TIE + Jancey-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop7 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed7 = std::chrono::duration_cast<std::chrono::milliseconds>(stop7 - start7);

		cout << filename << ", " << "Weighted TIE + jancey k-means, " << mse << ", " << numIters << ", " << elapsed7.count() << endl;

	}

	//#####################################################################

	//Image 2
	//#####################################################################

	/* Read Image*/
	filename = "image/4.2.04.ppm";
	img = read_PPM(filename);

	cout << "Image 2, 4.2.04.ppm" << endl;

	/*For each K value, perform the 8 algorithms in succession on different images*/
	for(int i = 0; i < numColors; i++)
	{
		cout << "K = " << k[i] << endl;
		cout << "===================" << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/*Initialize RGB table for weighted algorithms*/
		table = calc_color_table(img);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		jancey(img, k[i], cluster, numIters, alpha, true, mse); /*Batch k-means*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

		cout << filename << ", " << "batch k-means, " << mse << ", " << numIters << ", " << elapsed.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start1 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		jancey(img, k[i], cluster, numIters, alpha, false, mse); /*Jancey k-means*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop1 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop1 - start1);

		cout << filename << ", " << "jancey k-means, " << mse << ", " << numIters << ", " << elapsed1.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start2 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		weighted_jancey(table, k[i], cluster, numIters, alpha, true, mse); /*Weighted Batch-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop2 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed2 = std::chrono::duration_cast<std::chrono::milliseconds>(stop2 - start2);

		cout << filename << ", " << "weighted batch k-means, " << mse << ", " << numIters << ", " << elapsed2.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start3 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		weighted_jancey(table, k[i], cluster, numIters, alpha, false, mse); /*Weighted Jancey-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop3 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed3 = std::chrono::duration_cast<std::chrono::milliseconds>(stop3 - start3);

		cout << filename << ", " << "weighted jancey k-means, " << mse << ", " << numIters << ", " << elapsed3.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start4 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		TIE_jancey(img, k[i], cluster, numIters, alpha, true, mse); /*TIE + Batch-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop4 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed4 = std::chrono::duration_cast<std::chrono::milliseconds>(stop4 - start4);

		cout << filename << ", " << "TIE + batch k-means, " << mse << ", " << numIters << ", " << elapsed4.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start5 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		TIE_jancey(img, k[i], cluster, numIters, alpha, false, mse); /*TIE + Jancey-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop5 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed5 = std::chrono::duration_cast<std::chrono::milliseconds>(stop5 - start5);

		cout << filename << ", " << "TIE + jancey k-means, " << mse << ", " << numIters << ", " << elapsed5.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start6 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		weighted_TIE_jancey(table, k[i], cluster, numIters, alpha, true, mse); /*Weighted TIE + Batch-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop6 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed6 = std::chrono::duration_cast<std::chrono::milliseconds>(stop6 - start6);

		cout << filename << ", " << "weighted TIE + batch k-means, " << mse << ", " << numIters << ", " << elapsed6.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start7 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		weighted_TIE_jancey(table, k[i], cluster, numIters, alpha, false, mse); /*Weighted TIE + Jancey-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop7 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed7 = std::chrono::duration_cast<std::chrono::milliseconds>(stop7 - start7);

		cout << filename << ", " << "Weighted TIE + jancey k-means, " << mse << ", " << numIters << ", " << elapsed7.count() << endl;

	}

	//#####################################################################

	//Image 3
	//#####################################################################

	/* Read Image*/
	filename = "image/4.2.07.ppm";
	img = read_PPM(filename);

	cout << "Image 3, 4.2.07.ppm" << endl;

	/*For each K value, perform the 8 algorithms in succession on different images*/
	for(int i = 0; i < numColors; i++)
	{
		cout << "K = " << k[i] << endl;
		cout << "===================" << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/*Initialize RGB table for weighted algorithms*/
		table = calc_color_table(img);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		jancey(img, k[i], cluster, numIters, alpha, true, mse); /*Batch k-means*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

		cout << filename << ", " << "batch k-means, " << mse << ", " << numIters << ", " << elapsed.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start1 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		jancey(img, k[i], cluster, numIters, alpha, false, mse); /*Jancey k-means*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop1 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop1 - start1);

		cout << filename << ", " << "jancey k-means, " << mse << ", " << numIters << ", " << elapsed1.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start2 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		weighted_jancey(table, k[i], cluster, numIters, alpha, true, mse); /*Weighted Batch-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop2 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed2 = std::chrono::duration_cast<std::chrono::milliseconds>(stop2 - start2);

		cout << filename << ", " << "weighted batch k-means, " << mse << ", " << numIters << ", " << elapsed2.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start3 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		weighted_jancey(table, k[i], cluster, numIters, alpha, false, mse); /*Weighted Jancey-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop3 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed3 = std::chrono::duration_cast<std::chrono::milliseconds>(stop3 - start3);

		cout << filename << ", " << "weighted jancey k-means, " << mse << ", " << numIters << ", " << elapsed3.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start4 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		TIE_jancey(img, k[i], cluster, numIters, alpha, true, mse); /*TIE + Batch-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop4 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed4 = std::chrono::duration_cast<std::chrono::milliseconds>(stop4 - start4);

		cout << filename << ", " << "TIE + batch k-means, " << mse << ", " << numIters << ", " << elapsed4.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start5 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		TIE_jancey(img, k[i], cluster, numIters, alpha, false, mse); /*TIE + Jancey-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop5 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed5 = std::chrono::duration_cast<std::chrono::milliseconds>(stop5 - start5);

		cout << filename << ", " << "TIE + jancey k-means, " << mse << ", " << numIters << ", " << elapsed5.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start6 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		weighted_TIE_jancey(table, k[i], cluster, numIters, alpha, true, mse); /*Weighted TIE + Batch-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop6 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed6 = std::chrono::duration_cast<std::chrono::milliseconds>(stop6 - start6);

		cout << filename << ", " << "weighted TIE + batch k-means, " << mse << ", " << numIters << ", " << elapsed6.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start7 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		weighted_TIE_jancey(table, k[i], cluster, numIters, alpha, false, mse); /*Weighted TIE + Jancey-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop7 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed7 = std::chrono::duration_cast<std::chrono::milliseconds>(stop7 - start7);

		cout << filename << ", " << "Weighted TIE + jancey k-means, " << mse << ", " << numIters << ", " << elapsed7.count() << endl;

	}

	//#####################################################################

	//Image 4
	//#####################################################################

	/* Read Image*/
	filename = "image/fish.ppm";
	img = read_PPM(filename);

	cout << "Image 4, fish.ppm" << endl;

	/*For each K value, perform the 8 algorithms in succession on different images*/
	for(int i = 0; i < numColors; i++)
	{
		cout << "K = " << k[i] << endl;
		cout << "===================" << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/*Initialize RGB table for weighted algorithms*/
		table = calc_color_table(img);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		jancey(img, k[i], cluster, numIters, alpha, true, mse); /*Batch k-means*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

		cout << filename << ", " << "batch k-means, " << mse << ", " << numIters << ", " << elapsed.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start1 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		jancey(img, k[i], cluster, numIters, alpha, false, mse); /*Jancey k-means*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop1 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop1 - start1);

		cout << filename << ", " << "jancey k-means, " << mse << ", " << numIters << ", " << elapsed1.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start2 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		weighted_jancey(table, k[i], cluster, numIters, alpha, true, mse); /*Weighted Batch-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop2 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed2 = std::chrono::duration_cast<std::chrono::milliseconds>(stop2 - start2);

		cout << filename << ", " << "weighted batch k-means, " << mse << ", " << numIters << ", " << elapsed2.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start3 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		weighted_jancey(table, k[i], cluster, numIters, alpha, false, mse); /*Weighted Jancey-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop3 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed3 = std::chrono::duration_cast<std::chrono::milliseconds>(stop3 - start3);

		cout << filename << ", " << "weighted jancey k-means, " << mse << ", " << numIters << ", " << elapsed3.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start4 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		TIE_jancey(img, k[i], cluster, numIters, alpha, true, mse); /*TIE + Batch-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop4 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed4 = std::chrono::duration_cast<std::chrono::milliseconds>(stop4 - start4);

		cout << filename << ", " << "TIE + batch k-means, " << mse << ", " << numIters << ", " << elapsed4.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start5 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		TIE_jancey(img, k[i], cluster, numIters, alpha, false, mse); /*TIE + Jancey-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop5 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed5 = std::chrono::duration_cast<std::chrono::milliseconds>(stop5 - start5);

		cout << filename << ", " << "TIE + jancey k-means, " << mse << ", " << numIters << ", " << elapsed5.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start6 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		weighted_TIE_jancey(table, k[i], cluster, numIters, alpha, true, mse); /*Weighted TIE + Batch-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop6 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed6 = std::chrono::duration_cast<std::chrono::milliseconds>(stop6 - start6);

		cout << filename << ", " << "weighted TIE + batch k-means, " << mse << ", " << numIters << ", " << elapsed6.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start7 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		weighted_TIE_jancey(table, k[i], cluster, numIters, alpha, false, mse); /*Weighted TIE + Jancey-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop7 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed7 = std::chrono::duration_cast<std::chrono::milliseconds>(stop7 - start7);

		cout << filename << ", " << "Weighted TIE + jancey k-means, " << mse << ", " << numIters << ", " << elapsed7.count() << endl;

	}

	//#####################################################################

	//Image 5
	//#####################################################################

	/* Read Image*/
	filename = "image/goldhill.ppm";
	img = read_PPM(filename);

	cout << "Image 5, goldhill.ppm" << endl;

	/*For each K value, perform the 8 algorithms in succession on different images*/
	for(int i = 0; i < numColors; i++)
	{
		cout << "K = " << k[i] << endl;
		cout << "===================" << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/*Initialize RGB table for weighted algorithms*/
		table = calc_color_table(img);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		jancey(img, k[i], cluster, numIters, alpha, true, mse); /*Batch k-means*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

		cout << filename << ", " << "batch k-means, " << mse << ", " << numIters << ", " << elapsed.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start1 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		jancey(img, k[i], cluster, numIters, alpha, false, mse); /*Jancey k-means*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop1 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop1 - start1);

		cout << filename << ", " << "jancey k-means, " << mse << ", " << numIters << ", " << elapsed1.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start2 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		weighted_jancey(table, k[i], cluster, numIters, alpha, true, mse); /*Weighted Batch-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop2 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed2 = std::chrono::duration_cast<std::chrono::milliseconds>(stop2 - start2);

		cout << filename << ", " << "weighted batch k-means, " << mse << ", " << numIters << ", " << elapsed2.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start3 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		weighted_jancey(table, k[i], cluster, numIters, alpha, false, mse); /*Weighted Jancey-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop3 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed3 = std::chrono::duration_cast<std::chrono::milliseconds>(stop3 - start3);

		cout << filename << ", " << "weighted jancey k-means, " << mse << ", " << numIters << ", " << elapsed3.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start4 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		TIE_jancey(img, k[i], cluster, numIters, alpha, true, mse); /*TIE + Batch-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop4 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed4 = std::chrono::duration_cast<std::chrono::milliseconds>(stop4 - start4);

		cout << filename << ", " << "TIE + batch k-means, " << mse << ", " << numIters << ", " << elapsed4.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start5 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		TIE_jancey(img, k[i], cluster, numIters, alpha, false, mse); /*TIE + Jancey-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop5 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed5 = std::chrono::duration_cast<std::chrono::milliseconds>(stop5 - start5);

		cout << filename << ", " << "TIE + jancey k-means, " << mse << ", " << numIters << ", " << elapsed5.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start6 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		weighted_TIE_jancey(table, k[i], cluster, numIters, alpha, true, mse); /*Weighted TIE + Batch-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop6 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed6 = std::chrono::duration_cast<std::chrono::milliseconds>(stop6 - start6);

		cout << filename << ", " << "weighted TIE + batch k-means, " << mse << ", " << numIters << ", " << elapsed6.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start7 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		weighted_TIE_jancey(table, k[i], cluster, numIters, alpha, false, mse); /*Weighted TIE + Jancey-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop7 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed7 = std::chrono::duration_cast<std::chrono::milliseconds>(stop7 - start7);

		cout << filename << ", " << "Weighted TIE + jancey k-means, " << mse << ", " << numIters << ", " << elapsed7.count() << endl;

	}

	//#####################################################################

	//Image 6
	//#####################################################################

	/* Read Image*/
	filename = "image/kodim05.ppm";
	img = read_PPM(filename);

	cout << "Image 6, kodim05.ppm" << endl;

	/*For each K value, perform the 8 algorithms in succession on different images*/
	for(int i = 0; i < numColors; i++)
	{
		cout << "K = " << k[i] << endl;
		cout << "===================" << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/*Initialize RGB table for weighted algorithms*/
		table = calc_color_table(img);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		jancey(img, k[i], cluster, numIters, alpha, true, mse); /*Batch k-means*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

		cout << filename << ", " << "batch k-means, " << mse << ", " << numIters << ", " << elapsed.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start1 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		jancey(img, k[i], cluster, numIters, alpha, false, mse); /*Jancey k-means*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop1 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop1 - start1);

		cout << filename << ", " << "jancey k-means, " << mse << ", " << numIters << ", " << elapsed1.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start2 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		weighted_jancey(table, k[i], cluster, numIters, alpha, true, mse); /*Weighted Batch-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop2 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed2 = std::chrono::duration_cast<std::chrono::milliseconds>(stop2 - start2);

		cout << filename << ", " << "weighted batch k-means, " << mse << ", " << numIters << ", " << elapsed2.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start3 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		weighted_jancey(table, k[i], cluster, numIters, alpha, false, mse); /*Weighted Jancey-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop3 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed3 = std::chrono::duration_cast<std::chrono::milliseconds>(stop3 - start3);

		cout << filename << ", " << "weighted jancey k-means, " << mse << ", " << numIters << ", " << elapsed3.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start4 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		TIE_jancey(img, k[i], cluster, numIters, alpha, true, mse); /*TIE + Batch-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop4 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed4 = std::chrono::duration_cast<std::chrono::milliseconds>(stop4 - start4);

		cout << filename << ", " << "TIE + batch k-means, " << mse << ", " << numIters << ", " << elapsed4.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start5 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		TIE_jancey(img, k[i], cluster, numIters, alpha, false, mse); /*TIE + Jancey-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop5 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed5 = std::chrono::duration_cast<std::chrono::milliseconds>(stop5 - start5);

		cout << filename << ", " << "TIE + jancey k-means, " << mse << ", " << numIters << ", " << elapsed5.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start6 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		weighted_TIE_jancey(table, k[i], cluster, numIters, alpha, true, mse); /*Weighted TIE + Batch-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop6 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed6 = std::chrono::duration_cast<std::chrono::milliseconds>(stop6 - start6);

		cout << filename << ", " << "weighted TIE + batch k-means, " << mse << ", " << numIters << ", " << elapsed6.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start7 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		weighted_TIE_jancey(table, k[i], cluster, numIters, alpha, false, mse); /*Weighted TIE + Jancey-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop7 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed7 = std::chrono::duration_cast<std::chrono::milliseconds>(stop7 - start7);

		cout << filename << ", " << "Weighted TIE + jancey k-means, " << mse << ", " << numIters << ", " << elapsed7.count() << endl;

	}

	//#####################################################################

	//Image 7
	//#####################################################################

	/* Read Image*/
	filename = "image/kodim23.ppm";
	img = read_PPM(filename);

	cout << "Image 7, kodim23.ppm" << endl;

	/*For each K value, perform the 8 algorithms in succession on different images*/
	for(int i = 0; i < numColors; i++)
	{
		cout << "K = " << k[i] << endl;
		cout << "===================" << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/*Initialize RGB table for weighted algorithms*/
		table = calc_color_table(img);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		jancey(img, k[i], cluster, numIters, alpha, true, mse); /*Batch k-means*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

		cout << filename << ", " << "batch k-means, " << mse << ", " << numIters << ", " << elapsed.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start1 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		jancey(img, k[i], cluster, numIters, alpha, false, mse); /*Jancey k-means*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop1 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop1 - start1);

		cout << filename << ", " << "jancey k-means, " << mse << ", " << numIters << ", " << elapsed1.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start2 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		weighted_jancey(table, k[i], cluster, numIters, alpha, true, mse); /*Weighted Batch-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop2 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed2 = std::chrono::duration_cast<std::chrono::milliseconds>(stop2 - start2);

		cout << filename << ", " << "weighted batch k-means, " << mse << ", " << numIters << ", " << elapsed2.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start3 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		weighted_jancey(table, k[i], cluster, numIters, alpha, false, mse); /*Weighted Jancey-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop3 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed3 = std::chrono::duration_cast<std::chrono::milliseconds>(stop3 - start3);

		cout << filename << ", " << "weighted jancey k-means, " << mse << ", " << numIters << ", " << elapsed3.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start4 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		TIE_jancey(img, k[i], cluster, numIters, alpha, true, mse); /*TIE + Batch-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop4 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed4 = std::chrono::duration_cast<std::chrono::milliseconds>(stop4 - start4);

		cout << filename << ", " << "TIE + batch k-means, " << mse << ", " << numIters << ", " << elapsed4.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start5 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		TIE_jancey(img, k[i], cluster, numIters, alpha, false, mse); /*TIE + Jancey-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop5 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed5 = std::chrono::duration_cast<std::chrono::milliseconds>(stop5 - start5);

		cout << filename << ", " << "TIE + jancey k-means, " << mse << ", " << numIters << ", " << elapsed5.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start6 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		weighted_TIE_jancey(table, k[i], cluster, numIters, alpha, true, mse); /*Weighted TIE + Batch-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop6 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed6 = std::chrono::duration_cast<std::chrono::milliseconds>(stop6 - start6);

		cout << filename << ", " << "weighted TIE + batch k-means, " << mse << ", " << numIters << ", " << elapsed6.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start7 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		weighted_TIE_jancey(table, k[i], cluster, numIters, alpha, false, mse); /*Weighted TIE + Jancey-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop7 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed7 = std::chrono::duration_cast<std::chrono::milliseconds>(stop7 - start7);

		cout << filename << ", " << "Weighted TIE + jancey k-means, " << mse << ", " << numIters << ", " << elapsed7.count() << endl;

	}

	//#####################################################################

	//Image 8
	//#####################################################################

	/* Read Image*/
	filename = "image/pills.ppm";
	img = read_PPM(filename);

	cout << "Image 8, pills.ppm" << endl;

	/*For each K value, perform the 8 algorithms in succession on different images*/
	for(int i = 0; i < numColors; i++)
	{
		cout << "K = " << k[i] << endl;
		cout << "===================" << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/*Initialize RGB table for weighted algorithms*/
		table = calc_color_table(img);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		jancey(img, k[i], cluster, numIters, alpha, true, mse); /*Batch k-means*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

		cout << filename << ", " << "batch k-means, " << mse << ", " << numIters << ", " << elapsed.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start1 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		jancey(img, k[i], cluster, numIters, alpha, false, mse); /*Jancey k-means*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop1 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop1 - start1);

		cout << filename << ", " << "jancey k-means, " << mse << ", " << numIters << ", " << elapsed1.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start2 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		weighted_jancey(table, k[i], cluster, numIters, alpha, true, mse); /*Weighted Batch-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop2 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed2 = std::chrono::duration_cast<std::chrono::milliseconds>(stop2 - start2);

		cout << filename << ", " << "weighted batch k-means, " << mse << ", " << numIters << ", " << elapsed2.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start3 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		weighted_jancey(table, k[i], cluster, numIters, alpha, false, mse); /*Weighted Jancey-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop3 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed3 = std::chrono::duration_cast<std::chrono::milliseconds>(stop3 - start3);

		cout << filename << ", " << "weighted jancey k-means, " << mse << ", " << numIters << ", " << elapsed3.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start4 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		TIE_jancey(img, k[i], cluster, numIters, alpha, true, mse); /*TIE + Batch-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop4 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed4 = std::chrono::duration_cast<std::chrono::milliseconds>(stop4 - start4);

		cout << filename << ", " << "TIE + batch k-means, " << mse << ", " << numIters << ", " << elapsed4.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start5 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		TIE_jancey(img, k[i], cluster, numIters, alpha, false, mse); /*TIE + Jancey-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop5 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed5 = std::chrono::duration_cast<std::chrono::milliseconds>(stop5 - start5);

		cout << filename << ", " << "TIE + jancey k-means, " << mse << ", " << numIters << ", " << elapsed5.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start6 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		weighted_TIE_jancey(table, k[i], cluster, numIters, alpha, true, mse); /*Weighted TIE + Batch-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop6 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed6 = std::chrono::duration_cast<std::chrono::milliseconds>(stop6 - start6);

		cout << filename << ", " << "weighted TIE + batch k-means, " << mse << ", " << numIters << ", " << elapsed6.count() << endl;

		/*Initialize centers using maximin*/
		cluster = maximin(img, k[i]);

		/* Start Timer*/
		/*Declare data types of start, stop, and elapsed*/
		std::chrono::high_resolution_clock::time_point start7 = std::chrono::high_resolution_clock::now();

		/*Run Clustering function*/
		weighted_TIE_jancey(table, k[i], cluster, numIters, alpha, false, mse); /*Weighted TIE + Jancey-kmeans*/

		/* Stop Timer*/
		std::chrono::high_resolution_clock::time_point stop7 = std::chrono::high_resolution_clock::now();

		/* Execution Time*/
		std::chrono::milliseconds elapsed7 = std::chrono::duration_cast<std::chrono::milliseconds>(stop7 - start7);

		cout << filename << ", " << "Weighted TIE + jancey k-means, " << mse << ", " << numIters << ", " << elapsed7.count() << endl;

	}

	free(cluster);

	return 0;
}

