
Conversation opened. 20 messages. All messages read.

Skip to content
Using University of Central Arkansas Mail with screen readers

6 of 6,032
Experiments
Inbox

Emre Celebi
Thu, Aug 17, 2:09 PM (7 days ago)
Harrison, The previous email thread got too long. So, I'm starting a new one. I ran your program on my Linux machine. The output did not make a lot of sense bec

Harrison Bounds
Thu, Aug 17, 2:26 PM (7 days ago)
This makes sense, I was wondering why the TIE weighted algorithms ran extremely quickly. Please let me know if there is anything else I need to work on at this

Emre Celebi
Thu, Aug 17, 3:54 PM (7 days ago)
It's a strange problem. Somehow the time measurement does not work for the fast algorithms. I feel that the timing variable may be getting overridden. Once I fi

Harrison Bounds
Thu, Aug 17, 4:22 PM (7 days ago)
Hmm that it odd. Keep me posted on if the memory fixes work. Thank you.

Emre Celebi
Thu, Aug 17, 10:38 PM (7 days ago)
I've fixed all memory issues. Now the memory checker cannot find any problems with memory management. I also found the cause of the timing problem. It's because

Harrison Bounds
Thu, Aug 17, 10:49 PM (7 days ago)
Sounds good!

Emre Celebi
Fri, Aug 18, 3:46 PM (6 days ago)
to me

I've shortened the main function substantially, simplifying the experimental setup. The program does not leak memory anymore. I started an experiment on my work laptop. It will probably take a few days to complete.

Dr. Celebi
--
M. Emre Celebi, Ph.D., Fellow SPIE
he/him/his
Professor and Chair
Department of Computer Science and Engineering
College of Natural Sciences and Mathematics
University of Central Arkansas
Phone: 501-852-0931
Homepage: http://faculty.uca.edu/ecelebi/
GS: http://scholar.google.com/citations?user=mUzfrV8AAAAJ&hl=en




Harrison Bounds
Fri, Aug 18, 5:18 PM (6 days ago)
Sounds good, I will take a look at the new version soon.

Emre Celebi
Attachments
Fri, Aug 18, 7:23 PM (6 days ago)
to me

The current code is attached.

I eliminated Experiment 1 and expanded the scope of Experiment 2. Now, for each algorithm, the program prints the MSE, # iterations, and average CPU time. 

100 images
4 colors {4, 16, 64, 256}
12 algorithms (BKM, JKM1.2, JKM1.4, JKM1.6, JKM1.8, JKM1.99, TWBKM, TWJKM1.2, TWJKM1.4, TWJKM1.6, TWJKM1.8, TWJKM1.99)
10 repetitions (for averaging CPU times)
=
Total = 48,000 algorithm executions

I shortened the names of some of the variables. I also converted all the new's to malloc/calloc's and delete's to free's.

I expect the experiments to take about 2 days. It's difficult to predict run times because on some images the algorithms converge rapidly (e.g., 30 iterations) while on others they converge very slowly (e.g., 300 iterations). This is a problem of k-means hardly anyone talks about (there is no reliable way to know how many iterations it will take until convergence)

Dr. Celebi

Dr. Celebi
--
M. Emre Celebi, Ph.D., Fellow SPIE
he/him/his
Professor and Chair
Department of Computer Science and Engineering
College of Natural Sciences and Mathematics
University of Central Arkansas
Phone: 501-852-0931
Homepage: http://faculty.uca.edu/ecelebi/
GS: http://scholar.google.com/citations?user=mUzfrV8AAAAJ&hl=en



 One attachment
  •  Scanned by Gmail

Harrison Bounds
Fri, Aug 18, 10:23 PM (6 days ago)
Thank you for the update. I have familiarized myself with the new code.

Emre Celebi
Fri, Aug 18, 10:30 PM (6 days ago)
Great. Let's hope that the program runs to completion and the results make sense. Dr. Celebi

Harrison Bounds
Fri, Aug 18, 10:31 PM (6 days ago)
I will be hoping. Please let me know when the program is finished.

Emre Celebi
Fri, Aug 18, 10:32 PM (6 days ago)
I will find out when I get back to my office hopefully on Monday. Dr. Celebi

Emre Celebi
Mon, Aug 21, 7:54 AM (3 days ago)
Harrison, The program has been running for about 64 hours. So far, 343 out of 400 combinations have been processed (100 images x 4 colors). The experiment shoul

Harrison Bounds
Mon, Aug 21, 9:18 AM (3 days ago)
Thank you for the update. I am glad it is still running.

Emre Celebi
Tue, Aug 22, 8:24 AM (2 days ago)
Harrison, The experiment is over. I will start analyzing the output file once I get in front of my computer. Dr. Celebi

Harrison Bounds
Tue, Aug 22, 9:14 AM (2 days ago)
Sounds good!

Emre Celebi
AttachmentsTue, Aug 22, 10:28 PM (2 days ago)
Harrison, I believe I am done with the preliminary analysis of the experimental data. We have three criteria: - MSE (only the 6 fast algorithms: TWBKM, TWJKM1.2

Harrison Bounds
Tue, Aug 22, 11:46 PM (2 days ago)
I will take a look at these. As of right now I have time to work on the paper every day, so it will be my top priority. I will be awaiting further instructions.

Emre Celebi
Wed, Aug 23, 5:18 PM (17 hours ago)
to me

Sounds good! 
--
M. Emre Celebi, Ph.D., Fellow SPIE
he/him/his
Professor and Chair
Department of Computer Science and Engineering
College of Natural Sciences and Mathematics
University of Central Arkansas
Phone: 501-852-0931
Homepage: http://faculty.uca.edu/ecelebi/
GS: http://scholar.google.com/citations?user=mUzfrV8AAAAJ&hl=en



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
#include <iomanip>

using namespace std;
using namespace std::chrono;

typedef unsigned char uchar;
typedef unsigned long ulong;
typedef unsigned int uint;

typedef struct
{
	double red, green, blue, weight;
} RGB_Pixel;

typedef struct
{
	uint width, height;
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
	fprintf(fp, "%u %u\n", img->width, img->height);
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
		copy[i].center.weight = orig[i].center.weight;
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
free_image(RGB_Image *img)
{
	free ( img->data );
	free ( img );
}

RGB_Image*
map_pixels(const RGB_Image* inImg, const RGB_Cluster* cluster, const int numColors)
{
	double deltaR, deltaG, deltaB, dist, minDist;
	int minIndex;
	RGB_Image* outImg;
	RGB_Pixel pixel;

	outImg = ( RGB_Image *) malloc ( sizeof ( RGB_Image ) );
	outImg->data = ( RGB_Pixel * ) malloc ( inImg->size * sizeof ( RGB_Pixel ) );
	outImg->height = inImg->height;
	outImg->width = inImg->width;
	outImg->size = inImg->size;

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
free_table(RGB_Table *table)
{
	free ( table->data );
	free ( table );
}



/* Jancey Algorithm */
void
jkm(const RGB_Image* img, const int numColors, RGB_Cluster* clusters, int *numIters, const double alpha, const bool isBatch, double *mse)
{
	int numChanges, minIndex, size;
	double deltaR, deltaG, deltaB, dist, minDist;

	int *member = ( int * ) calloc ( img->size, sizeof ( int ) );

	RGB_Pixel pixel;
	RGB_Cluster *temp = ( RGB_Cluster * ) malloc ( numColors * sizeof ( RGB_Cluster ) );
	
	*numIters = 0;

	do
	{
		numChanges = 0;
		*mse = 0.0;

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

			*mse += minDist;
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

		(*numIters)++;

		//cout << "Iteration " << *numIters << ": SSE = " << *sse << " [" << "# changes = " << numChanges << "]" << endl;

	} while (numChanges != 0);

	*mse /= img->size;

	free ( member );
	free ( temp );
}

/* Weighted Jancey algorithm */
void
wjkm(const RGB_Table* colorTable, const int numColors, RGB_Cluster* clusters, int *numIters, const double alpha, const bool isBatch, double *mse)
{
	int numChanges, minIndex;
	double deltaR, deltaG, deltaB, dist, minDist, weight;
	int *member = ( int * ) calloc ( colorTable->size, sizeof ( int ) );

	RGB_Pixel pixel;
	RGB_Cluster *temp = ( RGB_Cluster * ) malloc ( numColors * sizeof ( RGB_Cluster ) );
	
	*numIters = 0;

	do
	{
		numChanges = 0;
		*mse = 0.0;

		reset_centers(temp, numColors);

		for (int i = 0; i < colorTable->size; i++)
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

			*mse += (pixel.weight * minDist);
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

		(*numIters)++;

		//cout << "Iteration " << *numIters << ": MSE = " << *mse << " [" << "# changes = " << numChanges << "]" << endl;

	} while (numChanges != 0);

	free ( member );
	free ( temp );
}

/* Jancey accelerated using TIE (Triangle Equality Elimination) */
void
tjkm(const RGB_Image* img, const int numColors, RGB_Cluster* clusters, int *numIters, const double alpha, const bool isBatch, double *mse)
{
	int numChanges, minIndex, size, tempIndex, oldMem;
	double deltaR, deltaG, deltaB, dist, minDist, delta;

	RGB_Cluster clustI, clustJ;
	RGB_Pixel pixel;
	RGB_Cluster *temp = ( RGB_Cluster * ) malloc ( numColors * sizeof ( RGB_Cluster ) );

	int *member = ( int * ) calloc ( img->size, sizeof ( int ) );

	/*center to center distance array sorted in ascending order by index*/
	int **p = ( int ** ) malloc ( numColors * sizeof ( int * ) ); 
	/*2d array to store center to center distances*/
	double **ccDist = ( double ** ) malloc ( numColors * sizeof ( double * ) ); 
	
	for (int i = 0; i < numColors; i++)
	{
		p[i] = ( int * ) malloc ( numColors * sizeof ( int ) );
		ccDist[i] = ( double * ) malloc ( numColors * sizeof ( double ) );
	}

	vector<pair<double, int>> vector;

	*numIters = 0;
	do
	{
		numChanges = 0;
		*mse = 0.0;

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

			*mse += minDist;
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

		(*numIters)++;

		//cout << "Iteration " << *numIters << ": SSE = " << *sse << " [" << "# changes = " << numChanges << "]" << endl;

	} while (numChanges != 0);

	*mse /= img->size;

	free ( member );
	free ( temp );

	for (int i = 0; i < numColors; i++)
	{
		free ( p[i] );
		free ( ccDist[i] );
	}
	free ( p );
	free ( ccDist );
}

void
twjkm(const RGB_Table* colorTable, const int numColors, RGB_Cluster* clusters, int *numIters, const double alpha, const bool isBatch, double *mse)
{
	int numChanges, minIndex, tempIndex, oldMem;
	double deltaR, deltaG, deltaB, dist, minDist, delta, weight;

	RGB_Cluster clustI, clustJ;
	RGB_Pixel pixel;
	RGB_Cluster *temp = ( RGB_Cluster * ) malloc ( numColors * sizeof ( RGB_Cluster ) );

	int *member = ( int * ) calloc ( colorTable->size, sizeof ( int ) );

	/*center to center distance array sorted in ascending order by index*/
	int **p = ( int ** ) malloc ( numColors * sizeof ( int * ) ); 
	/*2d array to store center to center distances*/
	double **ccDist = ( double ** ) malloc ( numColors * sizeof ( double * ) ); 
	
	for (int i = 0; i < numColors; i++)
	{
		p[i] = ( int * ) malloc ( numColors * sizeof ( int ) );
		ccDist[i] = ( double * ) malloc ( numColors * sizeof ( double ) );
	}

	vector<pair<double, int>> vector;

	*numIters = 0;
	do
	{
		numChanges = 0;
		*mse = 0.0;

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

		for (int i = 0; i < colorTable->size; i++)
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

			*mse += (pixel.weight * minDist);
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

		(*numIters)++;

		//cout << "Iteration " << *numIters << ": MSE = " << *mse << " [" << "# changes = " << numChanges << "]" << endl;

	} while (numChanges != 0);

	free ( member );
	free ( temp );
	
	for (int i = 0; i < numColors; i++)
	{
		free ( p[i] );
		free ( ccDist[i] );
	}
	free ( p );
	free ( ccDist );
}

/*Maximin algorithm to initialize k-means*/
RGB_Cluster *
maximin(const RGB_Image* img, const int num_colors)
	{	
		int num_pixels = img->size;
		RGB_Cluster *cluster;
		RGB_Cluster centroid;
		RGB_Pixel pixel;
		double red_sum = 0.0, green_sum = 0.0, blue_sum = 0.0;
		double delta_red, delta_green, delta_blue;
		double dist;
		double max_dist;
		int next_cluster;

		cluster = ( RGB_Cluster * ) malloc ( num_colors * sizeof ( RGB_Cluster ) );
		double *d = ( double * ) malloc ( num_pixels * sizeof ( double ) );

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
				free(d);
				return cluster;
	}


int 
main(int argc, char* argv[])
{
	const int numColors = 4;
	const int numImages = 100;
	const int numReps = 10;
	int colors[numColors] = {4, 16, 64, 256};
	const char* filenames[numImages] = {"images/adirondack_chairs.ppm","images/astro_bodies.ppm","images/astronaut.ppm","images/balinese_dancer.ppm","images/ball_caps.ppm","images/birthday_baloons.ppm","images/bosnian_pine_needle.ppm","images/buggy.ppm","images/calaveras.ppm","images/carrots.ppm","images/chalk_pastels.ppm","images/chicken_dish.ppm","images/chili_peppers.ppm","images/clownfish.ppm","images/color_chart.ppm","images/color_checker.ppm","images/coloring_pencils.ppm","images/columbia_crew.ppm","images/common_jezebel.ppm","images/common_lantanas.ppm","images/cosmic_vista.ppm","images/craft_cards.ppm","images/crepe_paper.ppm","images/cruise_ship.ppm","images/curler.ppm","images/daisy_bouquet.ppm","images/daisy_poster.ppm","images/easter_egg_basket.ppm","images/easter_eggs.ppm","images/eastern_rosella.ppm","images/felt_ball_trivet.ppm","images/fishing_nets.ppm","images/floating_market.ppm","images/fruit_dessert.ppm","images/fruit_stand.ppm","images/fruits.ppm","images/fruits_veggies.ppm","images/german_hot_air_balloon.ppm","images/girl.ppm","images/gourds.ppm","images/grilled_food.ppm","images/hard_candy.ppm","images/italian_hot_air_balloon.ppm","images/jacksons_chameleon.ppm","images/king_penguin.ppm","images/king_vulture.ppm","images/kingfisher.ppm","images/korean_dancer.ppm","images/lights.ppm","images/macarons.ppm","images/macaws.ppm","images/malayan_banded_pitta.ppm","images/mandarin_ducks.ppm","images/mandarinfish.ppm","images/mangoes.ppm","images/marrakech_museum.ppm","images/maya_beach.ppm","images/medicine_packets.ppm","images/moroccan_babouches.ppm","images/motocross.ppm","images/motorcycle.ppm","images/mural.ppm","images/nylon_cords.ppm","images/paper_clips.ppm","images/peacock.ppm","images/pencils.ppm","images/pigments.ppm","images/pink_mosque.ppm","images/plushies.ppm","images/prickly_pears.ppm","images/puffin.ppm","images/race_car.ppm","images/red_eyed_tree_frog.ppm","images/red_knobbed_starfish.ppm","images/rescue_helicopter.ppm","images/rose_bouquet.ppm","images/sagami_temple.ppm","images/salad_bowl.ppm","images/schoolgirls.ppm","images/seattle_great_wheel.ppm","images/shawls.ppm","images/shopping_bags.ppm","images/siberian_tiger.ppm","images/skiers.ppm","images/spices.ppm","images/sports_bicycles.ppm","images/sun_parakeet.ppm","images/tablet.ppm","images/textile_market.ppm","images/trade_fair_tower.ppm","images/traffic.ppm","images/tulip_field.ppm","images/umbrellas.ppm","images/veggie_pizza.ppm","images/veggies.ppm","images/venetian_lagoon.ppm","images/vintage_cars.ppm","images/wooden_toys.ppm","images/wool_carder_bee.ppm","images/yasaka_pagoda.ppm"};
	int bkm_iters, jkm_iters_12, jkm_iters_14, jkm_iters_16, jkm_iters_18, jkm_iters_199;
	int twbkm_iters, twjkm_iters_12, twjkm_iters_14, twjkm_iters_16, twjkm_iters_18, twjkm_iters_199;
	double bkm_mse, jkm_mse_12, jkm_mse_14, jkm_mse_16, jkm_mse_18, jkm_mse_199;
	double twbkm_mse, twjkm_mse_12, twjkm_mse_14, twjkm_mse_16, twjkm_mse_18, twjkm_mse_199;
	double alp10 = 1.0, alp12 = 1.2, alp14 = 1.4, alp16 = 1.6, alp18 = 1.8, alp199 = 1.99;
	double bkm_time, jkm_time_12, jkm_time_14, jkm_time_16, jkm_time_18, jkm_time_199;
	double twbkm_time, twjkm_time_12, twjkm_time_14, twjkm_time_16, twjkm_time_18, twjkm_time_199;
	std::chrono::duration<double> elapsed_time;
	RGB_Image *in_img;
	RGB_Cluster *initCenters, *copyCenters;
	RGB_Table *table; 

	srand(time(NULL));

	for (int i  = 0; i < numImages; i++)
	{
		/* Read Image*/
		in_img = read_PPM(filenames[i]);
		table = calc_color_table(in_img);
		for (int j = 0; j < numColors; j++)
		{
			bkm_time = 0.0;
			jkm_time_12 = 0.0;
			jkm_time_14 = 0.0;
			jkm_time_16 = 0.0;
			jkm_time_18 = 0.0;
			jkm_time_199 = 0.0;
			twbkm_time = 0.0;
			twjkm_time_12 = 0.0;
			twjkm_time_14 = 0.0;
			twjkm_time_16 = 0.0;
			twjkm_time_18 = 0.0;
			twjkm_time_199 = 0.0;
				
			initCenters = maximin(in_img, colors[j]);
			
			for (int k = 0; k < numReps; k++)
			{
				/*BKM*/
				copyCenters = duplicate_centers ( initCenters, colors[j] );
				auto start = high_resolution_clock::now();
				jkm(in_img, colors[j], copyCenters, &bkm_iters, alp10, true, &bkm_mse);
				auto stop = high_resolution_clock::now();
				elapsed_time = stop - start;
				bkm_time += static_cast<double>(elapsed_time.count());
				free ( copyCenters );
				
				/*JKM alpha = 1.2*/
				copyCenters = duplicate_centers ( initCenters, colors[j] );
				start = high_resolution_clock::now();
				jkm(in_img, colors[j], copyCenters, &jkm_iters_12, alp12, false, &jkm_mse_12); 
				stop = high_resolution_clock::now();
				elapsed_time = stop - start;
				jkm_time_12 += static_cast<double>(elapsed_time.count());
				free ( copyCenters );
				
				/*JKM alpha = 1.4*/
				copyCenters = duplicate_centers ( initCenters, colors[j] );
				start = high_resolution_clock::now();
				jkm(in_img, colors[j], copyCenters, &jkm_iters_14, alp14, false, &jkm_mse_14); 
				stop = high_resolution_clock::now();
				elapsed_time = stop - start;
				jkm_time_14 += static_cast<double>(elapsed_time.count());
				free ( copyCenters );
				
				/*JKM alpha = 1.6*/
				copyCenters = duplicate_centers ( initCenters, colors[j] );
				start = high_resolution_clock::now();
				jkm(in_img, colors[j], copyCenters, &jkm_iters_16, alp16, false, &jkm_mse_16); 
				stop = high_resolution_clock::now();
				elapsed_time = stop - start;
				jkm_time_16 += static_cast<double>(elapsed_time.count());
				free ( copyCenters );
				
				/*JKM alpha = 1.8*/
				copyCenters = duplicate_centers ( initCenters, colors[j] );
				start = high_resolution_clock::now();
				jkm(in_img, colors[j], copyCenters, &jkm_iters_18, alp18, false, &jkm_mse_18); 
				stop = high_resolution_clock::now();
				elapsed_time = stop - start;
				jkm_time_18 += static_cast<double>(elapsed_time.count());
				free ( copyCenters );
				
				/*JKM alpha = 1.99*/
				copyCenters = duplicate_centers ( initCenters, colors[j] );
				start = high_resolution_clock::now();
				jkm(in_img, colors[j], copyCenters, &jkm_iters_199, alp199, false, &jkm_mse_199); 
				stop = high_resolution_clock::now();
				elapsed_time = stop - start;
				jkm_time_199 += static_cast<double>(elapsed_time.count());
				free ( copyCenters );

				/*TWBKM*/
				copyCenters = duplicate_centers ( initCenters, colors[j] );
				start = high_resolution_clock::now();
				twjkm(table, colors[j], copyCenters, &twbkm_iters, alp10, true, &twbkm_mse);
				stop = high_resolution_clock::now();
				elapsed_time = stop - start;
				twbkm_time += static_cast<double>(elapsed_time.count());
				free ( copyCenters );

				/*TWJKM alpha = 1.2*/
				copyCenters = duplicate_centers ( initCenters, colors[j] );
				start = high_resolution_clock::now();
				twjkm(table, colors[j], copyCenters, &twjkm_iters_12, alp12, false, &twjkm_mse_12);
				stop = high_resolution_clock::now();
				elapsed_time = stop - start;
				twjkm_time_12 += static_cast<double>(elapsed_time.count());
				free ( copyCenters );

				/*TWJKM alpha = 1.4*/
				copyCenters = duplicate_centers ( initCenters, colors[j] );
				start = high_resolution_clock::now();
				twjkm(table, colors[j], copyCenters, &twjkm_iters_14, alp14, false, &twjkm_mse_14);
				stop = high_resolution_clock::now();
				elapsed_time = stop - start;
				twjkm_time_14 += static_cast<double>(elapsed_time.count());
				free ( copyCenters );

				/*TWJKM alpha = 1.6*/
				copyCenters = duplicate_centers ( initCenters, colors[j] );
				start = high_resolution_clock::now();
				twjkm(table, colors[j], copyCenters, &twjkm_iters_16, alp16, false, &twjkm_mse_16);
				stop = high_resolution_clock::now();
				elapsed_time = stop - start;
				twjkm_time_16 += static_cast<double>(elapsed_time.count());
				free ( copyCenters );

				/*TWJKM alpha = 1.8*/
				copyCenters = duplicate_centers ( initCenters, colors[j] );
				start = high_resolution_clock::now();
				twjkm(table, colors[j], copyCenters, &twjkm_iters_18, alp18, false, &twjkm_mse_18);
				stop = high_resolution_clock::now();
				elapsed_time = stop - start;
				twjkm_time_18 += static_cast<double>(elapsed_time.count());
				free ( copyCenters );

				/*TWJKM alpha = 1.99*/
				copyCenters = duplicate_centers ( initCenters, colors[j] );
				start = high_resolution_clock::now();
				twjkm(table, colors[j], copyCenters, &twjkm_iters_199, alp199, false, &twjkm_mse_199);
				stop = high_resolution_clock::now();
				elapsed_time = stop - start;
				twjkm_time_199 += static_cast<double>(elapsed_time.count());
				free ( copyCenters );
			}

			bkm_time /= numReps;
			jkm_time_12 /= numReps;
			jkm_time_14 /= numReps;
			jkm_time_16 /= numReps;
			jkm_time_18 /= numReps;
			jkm_time_199 /= numReps;
			twbkm_time /= numReps;
			twjkm_time_12 /= numReps;
			twjkm_time_14 /= numReps;
			twjkm_time_16 /= numReps;
			twjkm_time_18 /= numReps;
			twjkm_time_199 /= numReps;

			/*OUTPUT*/
			cout << filenames[i] << ", " << colors[j] << ", " 
			<< "BKM"      << ", " << bkm_mse       << ", " << bkm_iters       << ", " << bkm_time       << ", " 
			<< "JKM12"    << ", " << jkm_mse_12    << ", " << jkm_iters_12    << ", " << jkm_time_12    << ", "
			<< "JKM14"    << ", " << jkm_mse_14    << ", " << jkm_iters_14    << ", " << jkm_time_14    << ", "
			<< "JKM16"    << ", " << jkm_mse_16    << ", " << jkm_iters_16    << ", " << jkm_time_16    << ", "
			<< "JKM18"    << ", " << jkm_mse_18    << ", " << jkm_iters_18    << ", " << jkm_time_18    << ", "
			<< "JKM199"   << ", " << jkm_mse_199   << ", " << jkm_iters_199   << ", " << jkm_time_199   << ", "
			<< "TWBKM"    << ", " << twbkm_mse     << ", " << twbkm_iters     << ", " << twbkm_time     << ", " 
			<< "TWJKM12"  << ", " << twjkm_mse_12  << ", " << twjkm_iters_12  << ", " << twjkm_time_12  << ", "
			<< "TWJKM14"  << ", " << twjkm_mse_14  << ", " << twjkm_iters_14  << ", " << twjkm_time_14  << ", "
			<< "TWJKM16"  << ", " << twjkm_mse_16  << ", " << twjkm_iters_16  << ", " << twjkm_time_16  << ", "
			<< "TWJKM18"  << ", " << twjkm_mse_18  << ", " << twjkm_iters_18  << ", " << twjkm_time_18  << ", "
			<< "TWJKM199" << ", " << twjkm_mse_199 << ", " << twjkm_iters_199 << ", " << twjkm_time_199 << endl;
			
			free ( initCenters );
		}
		free_image(in_img);
		free_table(table);
	}

	cout << "DONE" << endl;

	return 0;
}