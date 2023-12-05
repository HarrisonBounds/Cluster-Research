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

		int* pixel_nearest_center = new int[num_pixels];

		int nearest_center_index; 
		int temp_nearest_index;
		double sse; /*Variable to store SSE*/
		double delta_red, delta_green, delta_blue, delta_red_temp, delta_green_temp, delta_blue_temp;
		double nearest_center_distance, nearest_center_distance_temp;
		RGB_Pixel pixel;
		RGB_Cluster *temp_clusters;

		temp_clusters = ( RGB_Cluster * ) malloc ( num_colors * sizeof ( RGB_Cluster ) );

		/*Initialize pixel center array*/
		for(int i = 0; i < num_pixels; i++)
		{
			pixel_nearest_center[i] = 0;
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

					nearest_center_distance_temp = delta_red_temp * delta_red_temp + delta_green_temp * delta_green_temp + delta_blue_temp * delta_blue_temp;
					
					/*The temp nearest center is closer to the pixel than its current nearest center*/
					if(nearest_center_distance_temp < nearest_center_distance || 
					(nearest_center_distance_temp == nearest_center_distance) && (temp_nearest_index < nearest_center_index))
					{	
						/*Update nearest center information*/
						nearest_center_distance = nearest_center_distance_temp; /*Current nearest center distance*/
						nearest_center_index = temp_nearest_index; /*Current nearest center index*/
						j = 0; /*Reset search*/
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
