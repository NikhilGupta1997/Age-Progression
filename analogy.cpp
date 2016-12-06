#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <vector>                                                               
#include <numeric>     

using namespace cv;
using namespace std;

static int flag_color;
static int Attributes = 12; // Number of features considered for aging

int N_large = 5;
int N_small = 3;

Vec3b get_colors(int i) {
	int b,g,r;
	switch(i) {
		case 0 : {
			b = 255;
			g = 255;
			r = 255;
			break;
		}
		case 1 : {
			b = 255;
			g = 0;
			r = 0;
			break;
		}
		case 2 : {
			b = 0;
			g = 255;
			r = 0;
			break;
		}
		case 3 : {
			b = 0;
			g = 0;
			r = 255;
			break;
		}
		case 4 : {
			b = 255;
			g = 255;
			r = 0;
			break;
		}
		case 5 : {
			b = 255;
			g = 0;
			r = 255;
			break;
		}
		case 6 : {
			b = 0;
			g = 255;
			r = 255;
			break;
		}
		case 7 : {
			b = 0;
			g = 128;
			r = 255;
			break;
		}
		case 8 : {
			b = 128;
			g = 0;
			r = 255;
			break;
		}
		case 9 : {
			b = 128;
			g = 255;
			r = 0;
			break;
		}
		case 10 : {
			b = 0;
			g = 0;
			r = 0;
			break;
		}
		case 11 : {
			b = 128;
			g = 128;
			r = 128;
			break;
		}
	}
	Vec3b pixel;
	pixel[0] = b;
	pixel[1] = g;
	pixel[2] = r;
	return pixel;
}

int get_component(Vec3b pixel) {
	Mat temp(1, 1, CV_8UC3, Scalar(0,0,0)); 
	temp.at<cv::Vec3b>(0) = pixel;
	cvtColor(temp, temp, CV_YUV2BGR);
	pixel = temp.at<cv::Vec3b>(0);
	// cout<<"pixel = "<<pixel<<endl;
	if(int(pixel[0]) > 215 && int(pixel[1]) > 215 && int(pixel[2]) > 215) { // White
		// cout<<"WHITE"<<endl;
		return 0;
	}
	else if(int(pixel[0]) > 215 && int(pixel[1]) < 30 && int(pixel[2]) < 30) { // Blue
		// cout<<"BLUE"<<endl;
		return 1;
	}
	else if(int(pixel[0]) < 30 && int(pixel[1]) > 215 && int(pixel[2]) < 30) { // Green
		// cout<<"GREEN"<<endl;
		return 2;
	}
	else if(int(pixel[0]) < 30 && int(pixel[1]) < 30 && int(pixel[2]) > 215) { // Red
		// cout<<"RED"<<endl;
		return 3;
	}
	else if(int(pixel[0]) > 215 && int(pixel[1]) > 215 && int(pixel[2]) < 30) { // Cyan 
		// cout<<"CYAN"<<endl;
		return 4;
	}
	else if(int(pixel[0]) > 215 && int(pixel[1]) < 30 && int(pixel[2]) > 215) { // Magenta
		// cout<<"MAGENTA"<<endl;
		return 5;
	}
	else if(int(pixel[0]) < 40 && int(pixel[1]) > 200 && int(pixel[2]) > 200) { // Yellow
		// cout<<"YELLOW"<<endl;
		return 6;
	}
	else if(int(pixel[0]) < 30 && int(pixel[1]) > 100 && int(pixel[2]) > 215) { // Orange
		// cout<<"ORANGE"<<endl;
		return 7;
	}
	else if(int(pixel[0]) >100 && int(pixel[1]) < 30 && int(pixel[2]) > 215) { // Mahroon
		// cout<<"MAHROON"<<endl;
		return 8;
	}
	else if(int(pixel[0]) >100 && int(pixel[1]) > 215 && int(pixel[2]) < 30) { // Green Blue
		// cout<<"GREEN-BLUE"<<endl;
		return 9;
	}
	else if(int(pixel[0]) > 105 && int(pixel[1]) > 105 && int(pixel[2]) > 105 && int(pixel[0]) < 135 && int(pixel[1]) < 135 && int(pixel[2]) < 135) { // Grey
		// cout<<"GREY"<<endl;
		return 11;
	}
	else {
		// cout<<"BACKGROUND"<<endl;
		return 10;
	}
}

void view_image(const Mat& image, string str) {
	namedWindow( str, CV_WINDOW_AUTOSIZE );// Create a window for display.
	imshow( str, image);
}

void wait_view_image(const Mat& image, string str) {
	namedWindow( str, CV_WINDOW_AUTOSIZE );// Create a window for display.
	imshow( str, image);
	waitKey(0);
}

// Function used to map the gradient of any part of one image to the corresponding part of the other image
Mat map_gradient(const Mat& old_changes, const Mat& old_outline, const Mat& new_outline, int num) {
	vector<int> x1, x2;
	vector<int> y1, y2;
	int x1_min = INT_MAX, x2_min = INT_MAX, y1_min = INT_MAX, y2_min = INT_MAX;
	int x1_max = INT_MIN, x2_max = INT_MIN, y1_max = INT_MIN, y2_max = INT_MIN;

	Mat outline = old_outline.clone();

	for(int i = 0; i < outline.rows; i++) {
		for(int j = 0; j < outline.cols; j++) {
			if(outline.at<uchar>(i,j) > 0) {
				int count = 0;
				int add = 0;
				if(i < outline.rows - 1 && outline.at<uchar>(i+1,j) == 0) {
					count++;
				}
				if(i > 0 && outline.at<uchar>(i-1,j) == 0) {
					count++;
				}
				if(j < outline.cols - 1 && outline.at<uchar>(i,j+1) == 0) {
					count++;
				}
				if(j > 0 && outline.at<uchar>(i-1,j) == 0) {
					count++;
				}
				if(count == 4)
					outline.at<uchar>(i,j) = 0;
			}
		}
	}

	view_image(outline, "outline");

	for(int i = 0; i < outline.rows; i++) {
		for(int j = 0; j < outline.cols; j++) {
			if(outline.at<uchar>(i,j) > 0) {
				x1.push_back(i);
				y1.push_back(j);
				x1_min = min(x1_min, i);
				y1_min = min(y1_min, j);
				x1_max = max(x1_max, i);
				y1_max = max(y1_max, j);
			}
		}
	}
	for(int i = 0; i < new_outline.rows; i++) {
		for(int j = 0; j < new_outline.cols; j++) {
			if(new_outline.at<uchar>(i,j) > 0) {
				x2.push_back(i);
				y2.push_back(j);
				x2_min = min(x2_min, i);
				y2_min = min(y2_min, j);
				x2_max = max(x2_max, i);
				y2_max = max(y2_max, j);
			}
		}
	}

	int average_x1 = accumulate( x1.begin(), x1.end(), 0.0)/x1.size(); 
	int average_y1 = accumulate( y1.begin(), y1.end(), 0.0)/y1.size(); 
	int average_x2 = accumulate( x2.begin(), x2.end(), 0.0)/x2.size(); 
	int average_y2 = accumulate( y2.begin(), y2.end(), 0.0)/y2.size(); 

	Mat output(new_outline.rows, new_outline.cols, CV_8UC1, Scalar(0));

	int x1_wid = fabs(x1_max - x1_min);
	int y1_wid = fabs(y1_max - y1_min);
	int x2_wid = fabs(x2_max - x2_min);
	int y2_wid = fabs(y2_max - y2_min);

	cout<<x1_wid<<" "<<y1_wid<<" "<<x2_wid<<" "<<y2_wid<<endl;

	for(int i = x1_min; i <= x1_max; i++) {
		for(int j = y1_min; j <= y1_max; j++) {
			if(old_changes.at<uchar>(i,j) > 0) {
				int xa_d = i - average_x1 ; 
				int ya_d = j - average_y1 ;
	           	int rb = average_x2 + xa_d*(float)((float)x2_wid/(float)x1_wid);
	           	int cb = average_y2 + ya_d*(float)((float)y2_wid/(float)y1_wid);
	           	if(rb < new_outline.rows && cb < new_outline.cols)
	           		output.at<uchar>(floor(rb),floor(cb)) = old_changes.at<uchar>(i,j);
			}
		}
	}
	return output;
}

// Function to obtain gradient of an input image
Mat get_gradient(const Mat& image) {
	Mat output, img_gray;
	Mat A_gradient, B_gradient, A_prime_gradient;
	cvtColor(image, img_gray, CV_BGR2GRAY);
	GaussianBlur( img_gray, img_gray, Size(3,3), 0, 0, BORDER_DEFAULT );

	int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    Scharr( img_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );
    Scharr( img_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );

    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, output );
    return output;
}

// Obtain the gradient corresponding to a certain feature, eg. forehead
Mat part_gradient(int num, const Mat& colored, const Mat& gradient) {
    Mat parts;
    parts = gradient.clone();
    for(int i = 0; i < colored.rows; i++) {
    	for(int j = 0; j < colored.cols; j++) {
    		if(get_component(colored.at<cv::Vec3b>(i,j)) == num && gradient.at<uchar>(i,j) > 0)
    			parts.at<uchar>(i,j) = gradient.at<uchar>(i,j);
    		else
    			parts.at<uchar>(i,j) = 0;
    	}
    }
    return parts;
}

// Obtain the outline of any feature
Mat outline_gradient(int num, const Mat& colored, const Mat& gradient) {
    Mat outline;
    outline = gradient.clone();
    for(int i = 0; i < colored.rows; i++) {
    	for(int j = 0; j < colored.cols; j++) {
    		if(get_component(colored.at<cv::Vec3b>(i,j)) == num && gradient.at<uchar>(i,j) > 10)
    			outline.at<uchar>(i,j) = gradient.at<uchar>(i,j);
    		else
    			outline.at<uchar>(i,j) = 0;
    	}
    }
    return outline;
}

// Returns the mapped gradient for a feature from one image to another
Mat get_mapped_part(int num, const Mat& A_color, const Mat& B_color, const Mat& A_grad, const Mat& B_grad, const Mat& changes) {
	Mat part_old = part_gradient(num, A_color, changes);
	Mat outline_old = outline_gradient(num, A_color, A_grad);
	Mat outline_new = outline_gradient(num, B_color, B_grad);
	Mat part_new = map_gradient(part_old, outline_old, outline_new, num);
	view_image(part_old, "part");
    view_image(outline_old, "outline_old");
    view_image(outline_new, "outline_new");
    view_image(part_new, "part_mapped");
	return part_new;
}

// Used to manage the indicies during parsing pixel and its neighborhood
void get_indices(int &start_i, int &end_i, int &start_j, int &end_j, int &pad_top, int &pad_bot, int &pad_left, int &pad_right, int &flag, int i, int j, int N, int h, int w) {
	int border = floor(N/2);

	int i_top = i-border;
	pad_top = 0;
	if(i_top < 0) {
	  start_i = 0;
	  pad_top =  -i_top;
	}
	else 
	  start_i = i_top;

	int i_bot = i+border;
	pad_bot = N-1;
	if(i_bot >= h) {
	  pad_bot = N-1-(i_bot-h);
	  end_i = h-1;
	}
	else
	  end_i = i_bot;

	int j_left = j-border;
	pad_left = 0;
	if(j_left < 0) {
	  start_j = 0;
	  pad_left = -j_left;
	}
	else
	  start_j = j_left;

	int j_right = j+border;
	pad_right = N-1;
	if(j_right >= w) {
	  end_j = w-1;
	  pad_right = N-1-(j_right-w);
	}
	else
	  end_j = j_right;

	flag = false;
}

// Construcs a feature vector for each pixel in the image
Mat concat_feature(vector<Mat> &X_pyramid, vector<Mat> &X_prime_pyramid, int l, int i, int j, int L) {
	Mat X_fine = X_pyramid[l];
	Mat X_prime_fine = X_prime_pyramid[l];
	int x_fine_height = X_fine.rows;
	int x_fine_width = X_fine.cols;

	Mat X_fine_nhood(N_large, N_large, CV_8UC3, Scalar(0,0,0));
	Mat X_prime_fine_nhood(N_large, N_large, CV_8UC3, Scalar(0,0,0));
	Mat X_coarse_nhood(N_small, N_small, CV_8UC3, Scalar(0,0,0));
	Mat X_prime_coarse_nhood(N_small, N_small, CV_8UC3, Scalar(0,0,0));

	int x_fine_start_i, x_fine_end_i, x_fine_start_j, x_fine_end_j, pad_top, pad_bot, pad_left, pad_right, flag;
	get_indices(x_fine_start_i, x_fine_end_i, x_fine_start_j, x_fine_end_j, pad_top, pad_bot, pad_left, pad_right, flag, i, j, N_large, x_fine_height, x_fine_width);

	for(int m = 0; m <= pad_bot-pad_top; m++) {
		for(int n = 0; n <= pad_right-pad_left; n++) {
			X_fine_nhood.at<cv::Vec3b>(pad_top + m, pad_bot + n) = X_fine.at<cv::Vec3b>(x_fine_start_i + m, x_fine_start_j + n);
		}
	}
	
	for(int m = 0; m <= pad_bot-pad_top; m++) {
		for(int n = 0; n <= pad_right-pad_left; n++) {
			X_prime_fine_nhood.at<cv::Vec3b>(pad_top + m, pad_bot + n) = X_prime_fine.at<cv::Vec3b>(x_fine_start_i + m, x_fine_start_j + n);
		}
	}

	if(l+1 <= L) {
		Mat X_coarse = X_pyramid[l];
		Mat X_prime_coarse = X_prime_pyramid[l];
		int x_coarse_height = X_coarse.rows;
		int x_coarse_width = X_coarse.cols;

		int x_coarse_start_i, x_coarse_end_i, x_coarse_start_j, x_coarse_end_j;
		get_indices(x_coarse_start_i, x_coarse_end_i, x_coarse_start_j, x_coarse_end_j, pad_top, pad_bot, pad_left, pad_right, flag, floor(i/2), floor(j/2), N_small, x_coarse_height, x_coarse_width);

		for(int m = 0; m <= pad_bot-pad_top; m++) {
			for(int n = 0; n <= pad_right-pad_left; n++) {
				X_coarse_nhood.at<cv::Vec3b>(pad_top + m, pad_bot + n) = X_coarse.at<cv::Vec3b>(x_coarse_start_i + m, x_coarse_start_j + n);
			}
		}
		
		for(int m = 0; m <= pad_bot-pad_top; m++) {
			for(int n = 0; n <= pad_right-pad_left; n++) {
				X_prime_coarse_nhood.at<cv::Vec3b>(pad_top + m, pad_bot + n) = X_prime_coarse.at<cv::Vec3b>(x_coarse_start_i + m, x_coarse_start_j + n);
			}
		}
	}

	Mat F(1, N_large*N_large*2 + N_small*N_small*2, CV_8UC3, Scalar(0,0,0));
	Mat temp_mat = X_fine_nhood.reshape(0, 1);
	for(int m = 0; m < N_large*N_large; m++) {
		F.at<cv::Vec3b>(m) = temp_mat.at<cv::Vec3b>(m);
	}
	temp_mat = X_prime_fine_nhood.reshape(0, 1);
	for(int m = 0; m < floor(N_large)*N_large + ceil(N_large/2); m++) {
		F.at<cv::Vec3b>(m + N_large*N_large) = temp_mat.at<cv::Vec3b>(m);
	}
	if(l+1 <= L) {
		temp_mat = X_coarse_nhood.reshape(0, 1);
		for(int m = 0; m < N_small*N_small; m++) {
			F.at<cv::Vec3b>(m + N_large*N_large*2) = temp_mat.at<cv::Vec3b>(m);
		}
		temp_mat = X_prime_coarse_nhood.reshape(0, 1);
		for(int m = 0; m < N_small*N_small; m++) {
			F.at<cv::Vec3b>(m + N_large*N_large*2 + N_small*N_small) = temp_mat.at<cv::Vec3b>(m);
		}
	}
	return F;
}

// Used to find the best approximate match in A based on the feature vector for a pixel in B
int best_approximate_match(vector<Mat>* A_features, vector<Mat> &A_pyramid, vector<Mat> &B_pyramid, vector<Mat>* B_features, int l, int i, int j, int &best_app_i, int &best_app_j, vector<int>** A_colored_points, vector<Mat> &B_colored_pyramid) {
	int index = B_pyramid[l].cols*i + j;
	int mat_size = B_features[l][index].cols;
	Mat query_point(1, mat_size, CV_8UC3, Scalar(0,0,0));
	query_point = B_features[l][index];
	double min_so_far = INT_MAX;
	int min_idx = 0;
	int comp = -1;
	if(flag_color == 0) {
		for(int ii = 0; ii < A_features[l].size(); ii++) {
			double dist = 0;
			for(int jj = 0; jj < mat_size; jj++) {
				dist += pow((A_features[l][ii].at<cv::Vec3b>(jj)[0] - query_point.at<cv::Vec3b>(jj)[0]),2);
			}
			if(dist < min_so_far) {
	    	 	min_idx = ii;
	      		min_so_far = dist;
	      	}
	    }
	}
	else {
		comp = get_component(B_colored_pyramid[l].at<cv::Vec3b>(i,j));
		for(int ii = 0; ii < A_colored_points[l][comp].size(); ii++) {
			int map = A_colored_points[l][comp][ii];
			double dist = 0;
			for(int jj = 0; jj < mat_size; jj++) {
				dist += pow((A_features[l][map].at<cv::Vec3b>(jj)[0] - query_point.at<cv::Vec3b>(jj)[0]),2);
			}
			if(dist < min_so_far) {
	    	 	min_idx = map;
	      		min_so_far = dist;
	      	}
	    }
	}
    best_app_i = min_idx / A_pyramid[l].cols;
    best_app_j = min_idx % A_pyramid[l].cols;
    if(flag_color == 0)
    	return 1;
    else
    	return comp;
}

// Used to find the best coherence match in A based on the feature vector for a pixel in B
void best_coherence_match(vector<Mat> &A_pyramid, vector<Mat> &A_prime_pyramid, vector<Mat> &B_pyramid, vector<Mat> &B_prime_pyramid, vector<Mat> &s_pyramid, int l, int L, int i, int j, int &best_coh_i, int &best_coh_j) {
	int A_h = A_pyramid[l].rows;
	int A_w = A_pyramid[l].cols;
	int border_big = floor(N_large/2);
	Mat F_q = concat_feature(B_pyramid, B_prime_pyramid, l, i, j, L);

	int min_dist = INT_MAX;
	int r_star_i = -1;
	int r_star_j = -1;

	bool done = false;
	for(int m = i-border_big; m<= i+border_big; m++) {
	  	for(int n = j-border_big; n<= j+border_big; n++) {
	    	if(m == i && n == j) {
	      		done = true;
	      		break;
	  		}
			int s_i = s_pyramid[l].at<cv::Vec3b>(m,n)[0];
	    	int s_j = s_pyramid[l].at<cv::Vec3b>(m,n)[1];
	    	int s_flag = s_pyramid[l].at<cv::Vec3b>(m,n)[2];

	    	int F_sr_i = s_i + (i - m);
    		int F_sr_j = s_j + (j - n);

    		if(F_sr_i >= A_h || F_sr_i < 0 || F_sr_j >= A_w || F_sr_j < 0 || s_flag == 0)
    			continue;

    		Mat F_sr = concat_feature(A_pyramid, A_prime_pyramid, l, F_sr_i, F_sr_j, L);

    		double dist = 0;
    		int F_length = N_large*N_large + N_small*N_small;
    		for(int m = 0; m < F_length; m++) {
				dist += pow((F_sr.at<cv::Vec3b>(m)[0] - F_q.at<cv::Vec3b>(m)[0]),2);
			}
	    	if(dist < min_dist) {
		      	min_dist = dist;
		      	r_star_i = m;
		      	r_star_j = n;
		      	best_coh_i = F_sr_i;
		      	best_coh_j = F_sr_j;
		    }
		}
	  	if(done)
	    	break;
	}

	if(r_star_i == -1 || r_star_j == -1) {
	  	best_coh_i = -1;
	  	best_coh_j = -1;
	}
	return;
}

// Decides on best pixel match in A for a pixel in B
int best_match(vector<Mat> &A_pyramid, vector<Mat> &A_prime_pyramid, vector<Mat> &B_pyramid, vector<Mat> &B_prime_pyramid, vector<Mat> &s_pyramid, vector<Mat>* A_features, vector<Mat>* B_features, int l, int L, int i, int j, int &best_i, int &best_j, vector<int>** A_colored_points, vector<Mat> &B_colored_pyramid) {
	double K = 1;

	// Get best approximate match
	int best_app_i, best_app_j;
	int color = best_approximate_match(A_features, A_pyramid, B_pyramid, B_features, l, i, j, best_app_i, best_app_j, A_colored_points, B_colored_pyramid);
	// cout<<"A "<<best_app_i<<" "<<best_app_j<<endl;

	// Do only best approximate match for the first 4 rows
	if(i < 4 || j < 4 || i >= B_pyramid[l].rows-4 || j >= B_pyramid[l].cols-4) {
  		best_i = best_app_i;
  		best_j = best_app_j;
  		return color;
	}

	if(flag_color == 1) {
		best_i = best_app_i;
	  	best_j = best_app_j;
		return color;
	}
	
	// Get best coherence match
	int best_coh_i, best_coh_j;
	best_coherence_match(A_pyramid, A_prime_pyramid, B_pyramid, B_prime_pyramid, s_pyramid, l, L, i, j, best_coh_i, best_coh_j);
	// cout<<"C "<<best_coh_i<<" "<<best_coh_j<<endl;
	if (best_coh_i == -1 || best_coh_j == -1) {
	  best_i = best_app_i;
	  best_j = best_app_j;
	  cout<<"coherence is -1 "<<endl;
	  return color;
	}

	Mat F_p_app = concat_feature(A_pyramid, A_prime_pyramid, l, best_app_i, best_app_j, L);
	Mat F_p_coh = concat_feature(A_pyramid, A_prime_pyramid, l, best_coh_i, best_coh_j, L);
	Mat F_q = concat_feature(B_pyramid, B_prime_pyramid, l, i, j, L);

	double d_app = 0;
	double d_coh = 0;
	int F_length = N_large*N_large + N_small*N_small;

	for(int m = 0; m < F_length; m++) {
		d_app += pow((F_p_app.at<cv::Vec3b>(m)[0] - F_q.at<cv::Vec3b>(m)[0]),2);
		d_coh += pow((F_p_coh.at<cv::Vec3b>(m)[0] - F_q.at<cv::Vec3b>(m)[0]),2);
	}

	if (d_coh <= d_app * (1 + pow(2,(l - L))*K)) {
	  	best_i = best_coh_i;
	  	best_j = best_coh_j;
	}
	else {
	  	best_i = best_app_i;
	  	best_j = best_app_j;
	}
	return color;
}

// Function to construct a gaussian pyramid for an input image
vector<Mat> buildGaussianPyramid(const Mat& A, int limit) {
    vector<Mat> maskGaussianPyramid;
    int levels = limit;
    maskGaussianPyramid.clear();
    Mat currentImg;
    maskGaussianPyramid.push_back(A); //highest level
    currentImg = A;
    for (int l=1; l<levels+1; l++) {
        pyrDown(currentImg, currentImg, Size(currentImg.cols/2, currentImg.rows/2));
        maskGaussianPyramid.push_back(currentImg);
    }
    return maskGaussianPyramid;
}

// Maps the analogy between A and Aprime to image B to obtain Bprime
Mat create_image_analogy(const Mat& A, const Mat& A_prime, const Mat& A_colored, const Mat& B, const Mat& B_colored) {
	int B_height = B.rows;
	int B_width = B.cols;
	int A_height = A.rows;
	int A_width = A.cols;

	// Create luminance maps of the input images
	cvtColor(A,A,CV_BGR2YUV);
	cvtColor(A_prime,A_prime,CV_BGR2YUV );
	cvtColor(B,B,CV_BGR2YUV );
	cvtColor(A_colored,A_colored,CV_BGR2YUV );
	cvtColor(B_colored,B_colored,CV_BGR2YUV );
	
	// Form image pyramids
	int Levels = 4;
	Mat B_prime(B_height, B_width, CV_8UC3, Scalar(0,0,0));
	Mat s(B_height, B_width, CV_8UC3, Scalar(0,0,0));

	vector<Mat> changes_pyramid, A_pyramid, A_prime_pyramid, A_colored_pyramid, B_pyramid, B_prime_pyramid, B_colored_pyramid, s_pyramid, A_pyramid_extended, B_pyramid_extended, A_gradient_pyramid, B_gradient_pyramid, A_colored_gradient_pyramid, B_colored_gradient_pyramid, A_prime_gradient_pyramid;
	vector<Mat> eyes_pyramid, lips_pyramid, chin_pyramid, cheek_pyramid, undereye_pyramid, forehead_pyramid, hair_pyramid, teeth_pyramid, neck_pyramid, lowerneck_pyramid, upperlip_pyramid;

	A_pyramid = buildGaussianPyramid(A, Levels);
	A_prime_pyramid = buildGaussianPyramid(A_prime, Levels);
	A_colored_pyramid = buildGaussianPyramid(A_colored, Levels);
	B_pyramid = buildGaussianPyramid(B, Levels);
	B_prime_pyramid = buildGaussianPyramid(B_prime, Levels);
	B_colored_pyramid = buildGaussianPyramid(B_colored, Levels);
	s_pyramid = buildGaussianPyramid(s, Levels);

	// Extend the borders of the images
	for(int i = 0; i <= Levels; i++) {
		int borderSize = N_large/2;
		Mat temp_A;
		copyMakeBorder(A_pyramid[i], temp_A, borderSize, borderSize, borderSize, borderSize, BORDER_REPLICATE);
		A_pyramid_extended.push_back(temp_A);
	}
	for(int i = 0; i <= Levels; i++) {
		int borderSize = N_large/2;
		Mat temp_B;
		copyMakeBorder(B_pyramid[i], temp_B, borderSize, borderSize, borderSize, borderSize, BORDER_REPLICATE);
		B_pyramid_extended.push_back(temp_B);
	}

	Size newsize = A.size();
	newsize.width = N_large;
	newsize.height = N_large;

	// Lets make the feature vectors of the images
	Mat New_mat;
	vector<Mat> *A_features, *B_features;
	vector<int> **A_colored_points, **B_colored_points;
	A_features = new vector<Mat>[Levels + 1];
	B_features = new vector<Mat>[Levels + 1];
	A_colored_points = new vector<int>*[Levels + 1];
	for(int i = 0; i < Levels+1; i++) {
		A_colored_points[i] = new vector<int>[Attributes];
	}
	B_colored_points = new vector<int>*[Levels + 1];
	for(int i = 0; i < Levels+1; i++) {
		B_colored_points[i] = new vector<int>[Attributes];
	}
	for(int m = 0; m < A_pyramid.size(); m++) {
		for(int i = 0; i < A_pyramid[m].rows; i++) {
			for(int j = 0; j < A_pyramid[m].cols; j++) {
				Mat New_mat;
				New_mat.create(newsize, A_pyramid_extended[m].type());
				for(int k = 0; k < N_large; k++) {
					for(int l = 0; l < N_large; l++) {
						New_mat.at<cv::Vec3b>(k,l) = A_pyramid_extended[m].at<cv::Vec3b>(i+k,j+l);
					}
				}
				Mat temp_mat = New_mat.reshape(0, 1);
				A_features[m].push_back(temp_mat);
			}
		}
	}
	for(int m = 0; m < B_pyramid.size(); m++) {
		for(int i = 0; i < B_pyramid[m].rows; i++) {
			for(int j = 0; j < B_pyramid[m].cols; j++) {
				Mat New_mat;
				New_mat.create(newsize, B_pyramid_extended[m].type());
				for(int k = 0; k < N_large; k++) {
					for(int l = 0; l < N_large; l++) {
						New_mat.at<cv::Vec3b>(k,l) = B_pyramid_extended[m].at<cv::Vec3b>(i+k,j+l);
					}
				}
				Mat temp_mat = New_mat.reshape(0, 1);
				B_features[m].push_back(temp_mat);
			}
		}
	}

	// Create a 2D array of vectors of the colored points
	Mat A_old_image(A_colored_pyramid[0].rows, A_colored_pyramid[0].cols, CV_8UC3, Scalar(0,0,0));
	for(int m = 0; m < A_colored_pyramid.size(); m++) {
		for(int i = 0; i < A_colored_pyramid[m].rows; i++) {
			for(int j = 0; j < A_colored_pyramid[m].cols; j++) {
				int comp = get_component(A_colored_pyramid[m].at<cv::Vec3b>(i,j));
				if(m == 0) {
					A_old_image.at<cv::Vec3b>(i, j) = get_colors(comp);
				}
				A_colored_points[m][comp].push_back(i*A_colored_pyramid[m].cols + j);
			}
		}
	}

	view_image(A_old_image, "Anew_image");

	// view coloered parts of the colored image
	Mat B_old_image(B_colored_pyramid[0].rows, B_colored_pyramid[0].cols, CV_8UC3, Scalar(0,0,0));
	for(int i = 0; i < B_colored_pyramid[0].rows; i++) {
		for(int j = 0; j < B_colored_pyramid[0].cols; j++) {
			int comp = get_component(B_colored_pyramid[0].at<cv::Vec3b>(i,j));
			B_old_image.at<cv::Vec3b>(i, j) = get_colors(comp);
		}
	}

	view_image(B_old_image, "Bnew_image");

	// Get gradients
	Mat A_colored_gradient, B_colored_gradient;
	Mat A_gradient, B_gradient, A_prime_gradient;

	A_colored_gradient = get_gradient(A_old_image);
	B_colored_gradient = get_gradient(B_old_image);
	A_gradient = get_gradient(A);
	B_gradient = get_gradient(B);
	A_prime_gradient = get_gradient(A_prime);

	// Sharpen the gradients
    for(int i = 0; i < A_gradient.rows; i++) {
    	for(int j = 0; j < A_gradient.cols; j++) {
    		if(A_gradient.at<uchar>(i,j) > 10)
    			A_gradient.at<uchar>(i,j) *= 5;
    	}
    }

    for(int i = 0; i < A_prime_gradient.rows; i++) {
    	for(int j = 0; j < A_prime_gradient.cols; j++) {
    		if(A_prime_gradient.at<uchar>(i,j) > 10)
    			A_prime_gradient.at<uchar>(i,j) *= 5;
    	}
    }

    // Create gradient pyramids
	A_gradient_pyramid = buildGaussianPyramid(A_gradient, Levels);
	B_gradient_pyramid = buildGaussianPyramid(B_gradient, Levels);
	A_prime_gradient_pyramid = buildGaussianPyramid(A_prime_gradient, Levels);
	A_colored_gradient_pyramid = buildGaussianPyramid(A_colored_gradient, Levels);
	B_colored_gradient_pyramid = buildGaussianPyramid(B_colored_gradient, Levels);

	// Create a changes image to map the wrinkles
	Mat changes;
	changes = A_prime_gradient_pyramid[0] - A_gradient_pyramid[0];

	for(int i = 0; i < changes.rows; i++) {
    	for(int j = 0; j < changes.cols; j++) {
    		int the_comp = get_component(A_colored_pyramid[0].at<cv::Vec3b>(i,j));
    		if(the_comp != 2 && the_comp != 7 && the_comp != 1 && the_comp != 8 && the_comp != 9 && the_comp != 0 && the_comp != 5 && the_comp != 6)
    			changes.at<uchar>(i,j) = 0;
    		else if(changes.at<uchar>(i,j) > 5)
    			changes.at<uchar>(i,j) = 100;
    	}
    } 

    // Isolate the gradient changes of a certain part

    // Head - 2
	Mat part_new = get_mapped_part(2, A_colored_pyramid[0], B_colored_pyramid[0], A_colored_gradient, B_colored_gradient, changes);
    forehead_pyramid = buildGaussianPyramid(part_new, Levels);

    // Cheek - 1
	part_new = get_mapped_part(1, A_colored_pyramid[0], B_colored_pyramid[0], A_colored_gradient, B_colored_gradient, changes);
    cheek_pyramid = buildGaussianPyramid(part_new, Levels);

    // Undereye - 7
	part_new = get_mapped_part(7, A_colored_pyramid[0], B_colored_pyramid[0], A_colored_gradient, B_colored_gradient, changes);
    undereye_pyramid = buildGaussianPyramid(part_new, Levels);

    // Chin - 9
	part_new = get_mapped_part(9, A_colored_pyramid[0], B_colored_pyramid[0], A_colored_gradient, B_colored_gradient, changes);
    chin_pyramid = buildGaussianPyramid(part_new, Levels);

    // Lips - 6
	part_new = get_mapped_part(6, A_colored_pyramid[0], B_colored_pyramid[0], A_colored_gradient, B_colored_gradient, changes);
    lips_pyramid = buildGaussianPyramid(part_new, Levels);

    // Upper Lips - 8
	part_new = get_mapped_part(8, A_colored_pyramid[0], B_colored_pyramid[0], A_colored_gradient, B_colored_gradient, changes);
    upperlip_pyramid = buildGaussianPyramid(part_new, Levels);

    // Neck - 0
	part_new = get_mapped_part(0, A_colored_pyramid[0], B_colored_pyramid[0], A_colored_gradient, B_colored_gradient, changes);
    neck_pyramid = buildGaussianPyramid(part_new, Levels);

    // Lowerneck - 11
	part_new = get_mapped_part(11, A_colored_pyramid[0], B_colored_pyramid[0], A_colored_gradient, B_colored_gradient, changes);
    lowerneck_pyramid = buildGaussianPyramid(part_new, Levels);
    

    changes_pyramid = buildGaussianPyramid(changes, Levels);

	view_image(A_gradient_pyramid[0], "A_gradient_pyramid");

	view_image(A_prime_gradient_pyramid[0], "A_prime_gradient_pyramid");

	view_image(changes, "A_change_gradient_pyramid");

	view_image(B_gradient_pyramid[0], "B_gradient_pyramid");

	view_image(A_colored_gradient_pyramid[0], "A_colored_gradient_pyramid");

	view_image(B_colored_gradient_pyramid[0], "B_colored_gradient_pyramid");

	// Find the best match

	if(flag_color == 1) {
		int color_pixel;
		for(int l = Levels; l >= 0; l--) {
			int height_prime = B_prime_pyramid[l].rows;
			int width_prime = B_prime_pyramid[l].cols;
			int best_i, best_j;
			for(int i = 0; i < height_prime; i++) {
				for(int j = 0; j < width_prime; j++) {
					cout<<"Going for best match "<< i <<" "<<j<<endl;
					if(get_component(B_colored_pyramid[l].at<cv::Vec3b>(i,j)) == 10) {
						best_i = i;
						best_j = j;
						color_pixel = 10;
						if(B_pyramid[l].at<cv::Vec3b>(i, j)[0] > 10)
							B_prime_pyramid[l].at<cv::Vec3b>(i, j)[0] = B_pyramid[l].at<cv::Vec3b>(i, j)[0] - 10;
						else
							B_prime_pyramid[l].at<cv::Vec3b>(i, j)[0] = B_pyramid[l].at<cv::Vec3b>(i, j)[0];
						B_prime_pyramid[l].at<cv::Vec3b>(i, j)[1] = B_pyramid[l].at<cv::Vec3b>(i, j)[1];
						B_prime_pyramid[l].at<cv::Vec3b>(i, j)[2] = B_pyramid[l].at<cv::Vec3b>(i, j)[2];
					}
					else {
						color_pixel = best_match(A_pyramid, A_prime_pyramid, B_pyramid, B_prime_pyramid, s_pyramid, A_features, B_features, l, Levels, i, j, best_i, best_j, A_colored_points, B_colored_pyramid);
						s_pyramid[l].at<cv::Vec3b>(i,j)[0] = best_i;
						s_pyramid[l].at<cv::Vec3b>(i,j)[1] = best_j;
						s_pyramid[l].at<cv::Vec3b>(i,j)[2] = 1;
						
						// 0, 1, 2, 5, 6, 7, 8, 9
						// 2  7  1  8  9  0  5  6

						// Map luminence
						if(color_pixel == 0) {// neck
							if(neck_pyramid[l].at<uchar>(i,j) > 80) {
								B_prime_pyramid[l].at<cv::Vec3b>(i, j)[0] = B_pyramid[l].at<cv::Vec3b>(i, j)[0] + 10;// + 2*(A_prime_pyramid[l].at<cv::Vec3b>(best_i, best_j)[0] - A_pyramid[l].at<cv::Vec3b>(best_i, best_j)[0]));
								B_prime_pyramid[l].at<cv::Vec3b>(i-1, j)[0] -= 10;
							}
							B_prime_pyramid[l].at<cv::Vec3b>(i, j)[0] = (B_pyramid[l].at<cv::Vec3b>(i, j)[0] + 1*A_prime_pyramid[l].at<cv::Vec3b>(best_i, best_j)[0])/2;
						}
						else if(color_pixel == 1) // cheeks
							// if(get_component(B_colored_pyramid[l].at<Vec3b>(i-1,j)) == 1 && get_component(B_colored_pyramid[l].at<Vec3b>(i+1,j)) == 8 ){
							if(B_colored_gradient.at<uchar>(i,j) > 0 && !(B_colored_gradient.at<uchar>(i,j) > 0) && get_component(B_colored_pyramid[l].at<Vec3b>(i-1,j)) == 1 && get_component(B_colored_pyramid[l].at<Vec3b>(i+1,j)) != 1 ) {
								B_prime_pyramid[l].at<cv::Vec3b>(i, j)[0] = B_pyramid[l].at<cv::Vec3b>(i, j)[0] - 20;
								B_prime_pyramid[l].at<cv::Vec3b>(i-1, j)[0] = B_pyramid[l].at<cv::Vec3b>(i, j)[0] - 40;
								B_prime_pyramid[l].at<cv::Vec3b>(i+1, j)[0] = B_pyramid[l].at<cv::Vec3b>(i, j)[0]- 30;
							}
								//B_prime_pyramid[l].at<cv::Vec3b>(i, j)[0] = B_pyramid[l].at<cv::Vec3b>(i, j)[0] - 25;
							else if(cheek_pyramid[l].at<uchar>(i,j) > 80) {
								B_prime_pyramid[l].at<cv::Vec3b>(i, j)[0] = B_pyramid[l].at<cv::Vec3b>(i, j)[0] + 10;// + 2*(A_prime_pyramid[l].at<cv::Vec3b>(best_i, best_j)[0] - A_pyramid[l].at<cv::Vec3b>(best_i, best_j)[0]));
								B_prime_pyramid[l].at<cv::Vec3b>(i-1, j)[0] -= 10;
							}
							else
								B_prime_pyramid[l].at<cv::Vec3b>(i, j)[0] = B_pyramid[l].at<cv::Vec3b>(i, j)[0] - 5;
						else if(color_pixel == 2) // forehead
							if(forehead_pyramid[l].at<uchar>(i,j) > 80) {
								B_prime_pyramid[l].at<cv::Vec3b>(i, j)[0] = (B_pyramid[l].at<cv::Vec3b>(i, j)[0] + 5 + 3*(A_prime_pyramid[l].at<cv::Vec3b>(best_i, best_j)[0] - A_pyramid[l].at<cv::Vec3b>(best_i, best_j)[0]));
								B_prime_pyramid[l].at<cv::Vec3b>(i-1, j)[0] -= 10;
							}
							else
								B_prime_pyramid[l].at<cv::Vec3b>(i, j)[0] = B_pyramid[l].at<cv::Vec3b>(i, j)[0] - 5;
						else if(color_pixel == 3) { // hair
							if(get_component(B_colored_pyramid[l].at<Vec3b>(i-1,j)) == 2) {
								B_prime_pyramid[l].at<cv::Vec3b>(i-1, j)[0] += 20;
								B_prime_pyramid[l].at<cv::Vec3b>(i, j)[0] = (B_prime_pyramid[l].at<cv::Vec3b>(i-1, j)[0] + A_prime_pyramid[l].at<cv::Vec3b>(best_i, best_j)[0]) / 2;
							}
							B_prime_pyramid[l].at<cv::Vec3b>(i, j)[0] = 70 + B_pyramid[l].at<cv::Vec3b>(i, j)[0];//(A_prime_pyramid[l].at<cv::Vec3b>(best_i, best_j)[0] + 15);
						}
						else if(color_pixel == 4) // eyes
							B_prime_pyramid[l].at<cv::Vec3b>(i, j)[0] = B_pyramid[l].at<cv::Vec3b>(i, j)[0];
						else if(color_pixel == 5) // teeth
							B_prime_pyramid[l].at<cv::Vec3b>(i, j)[0] = B_pyramid[l].at<cv::Vec3b>(i, j)[0] - 20;
						else if(color_pixel == 6) {// lips
							if(lips_pyramid[l].at<uchar>(i,j) > 80) {
								B_prime_pyramid[l].at<cv::Vec3b>(i, j)[0] = B_pyramid[l].at<cv::Vec3b>(i, j)[0] + 10;// + 2*(A_prime_pyramid[l].at<cv::Vec3b>(best_i, best_j)[0] - A_pyramid[l].at<cv::Vec3b>(best_i, best_j)[0]));
								B_prime_pyramid[l].at<cv::Vec3b>(i, j-1)[0] -= 10;
							}
							B_prime_pyramid[l].at<cv::Vec3b>(i, j)[0] = (2*B_pyramid[l].at<cv::Vec3b>(i, j)[0] + (A_prime_pyramid[l].at<cv::Vec3b>(best_i, best_j)[0] - A_pyramid[l].at<cv::Vec3b>(best_i, best_j)[0]))/1.5;
						}
						else if(color_pixel == 7) {
							if(undereye_pyramid[l].at<uchar>(i,j) > 80) {
								B_prime_pyramid[l].at<cv::Vec3b>(i, j)[0] = B_pyramid[l].at<cv::Vec3b>(i, j)[0] + 14;// + 3*(A_prime_pyramid[l].at<cv::Vec3b>(best_i, best_j)[0] - A_pyramid[l].at<cv::Vec3b>(best_i, best_j)[0]));
								B_prime_pyramid[l].at<cv::Vec3b>(i-1, j)[0] -= 14;
							}
							else
								B_prime_pyramid[l].at<cv::Vec3b>(i, j)[0] = B_pyramid[l].at<cv::Vec3b>(i, j)[0] - 5;
						}
						else if(color_pixel == 8) {
							if(upperlip_pyramid[l].at<uchar>(i,j) > 80) 
								B_prime_pyramid[l].at<cv::Vec3b>(i, j)[0] = (B_pyramid[l].at<cv::Vec3b>(i, j)[0] + 10 + 2*(A_prime_pyramid[l].at<cv::Vec3b>(best_i, best_j)[0] - A_pyramid[l].at<cv::Vec3b>(best_i, best_j)[0]));
							else
								B_prime_pyramid[l].at<cv::Vec3b>(i, j)[0] = B_pyramid[l].at<cv::Vec3b>(i, j)[0] - 5;
						}
						else if(color_pixel == 9) { // chin
							if(chin_pyramid[l].at<uchar>(i,j) > 80) {
								B_prime_pyramid[l].at<cv::Vec3b>(i, j-1)[0] -= 20;
								B_prime_pyramid[l].at<cv::Vec3b>(i, j)[0] = (B_pyramid[l].at<cv::Vec3b>(i, j)[0] + 20 + (A_prime_pyramid[l].at<cv::Vec3b>(best_i, best_j)[0] - A_pyramid[l].at<cv::Vec3b>(best_i, best_j)[0]));
							}
							else
								B_prime_pyramid[l].at<cv::Vec3b>(i, j)[0] = B_pyramid[l].at<cv::Vec3b>(i, j)[0] - 5;
						}
						else if(color_pixel == 11) { //underchin
							if(lowerneck_pyramid[l].at<uchar>(i,j) > 80) {
								B_prime_pyramid[l].at<cv::Vec3b>(i-1, j)[0] -= 10;
								B_prime_pyramid[l].at<cv::Vec3b>(i, j)[0] = 2 + (B_pyramid[l].at<cv::Vec3b>(i, j)[0] + 2*(A_prime_pyramid[l].at<cv::Vec3b>(best_i, best_j)[0] - A_pyramid[l].at<cv::Vec3b>(best_i, best_j)[0]));
							}
							else
								B_prime_pyramid[l].at<cv::Vec3b>(i, j)[0] = 2 + (B_pyramid[l].at<cv::Vec3b>(i, j)[0] + B_pyramid[l].at<cv::Vec3b>(i, j+1)[0] + B_pyramid[l].at<cv::Vec3b>(i, j-1)[0] + B_pyramid[l].at<cv::Vec3b>(i+1, j)[0] + B_pyramid[l].at<cv::Vec3b>(i-1, j)[0] + (A_prime_pyramid[l].at<cv::Vec3b>(best_i, best_j)[0] - A_pyramid[l].at<cv::Vec3b>(best_i, best_j)[0]))/5;
						}

						if(B_colored_gradient_pyramid[l].at<uchar>(i,j) > 0) {
							int count = 0;
							int add = 0;
							if(i < B_colored_gradient_pyramid[l].rows - 1 && get_component(B_colored_pyramid[l].at<cv::Vec3b>(i+1, j)) != 10 && get_component(B_colored_pyramid[l].at<cv::Vec3b>(i-1, j)) != 3) {
								add += B_pyramid[l].at<cv::Vec3b>(i+1, j)[0];
								count++;
							}
							if(i > 0 && get_component(B_colored_pyramid[l].at<cv::Vec3b>(i-1, j)) != 10 && get_component(B_colored_pyramid[l].at<cv::Vec3b>(i-1, j)) != 3) {
								add += B_pyramid[l].at<cv::Vec3b>(i-1, j)[0];
								count++;
							}
							if(j < B_colored_gradient_pyramid[l].cols - 1 && get_component(B_colored_pyramid[l].at<cv::Vec3b>(i, j+1)) != 10 && get_component(B_colored_pyramid[l].at<cv::Vec3b>(i-1, j)) != 3) {
								add += B_pyramid[l].at<cv::Vec3b>(i, j+1)[0];
								count++;
							}
							if(j > 0 && get_component(B_colored_pyramid[l].at<cv::Vec3b>(i, j-1)) != 10 && get_component(B_colored_pyramid[l].at<cv::Vec3b>(i-1, j)) != 3) {
								add += B_pyramid[l].at<cv::Vec3b>(i, j-1)[0];
								count++;
							}
							B_prime_pyramid[l].at<cv::Vec3b>(i, j)[0] = (B_prime_pyramid[l].at<cv::Vec3b>(i, j)[0] + add)/(count + 1);
						}

						if(color_pixel != 4)
							B_prime_pyramid[l].at<cv::Vec3b>(i, j)[0] -= 10;

						// Map color
						if(color_pixel == 5) {
							B_prime_pyramid[l].at<cv::Vec3b>(i, j)[1] = B_pyramid[l].at<cv::Vec3b>(i, j)[1]+30;
							B_prime_pyramid[l].at<cv::Vec3b>(i, j)[2] = B_pyramid[l].at<cv::Vec3b>(i, j)[2]-15;
						}
						else if(color_pixel == 6) {
							B_prime_pyramid[l].at<cv::Vec3b>(i, j)[1] = 160;
							B_prime_pyramid[l].at<cv::Vec3b>(i, j)[2] = 105;
						}
						else if(color_pixel == 3) {
							B_prime_pyramid[l].at<cv::Vec3b>(i, j)[1] = 128 + 7;
							B_prime_pyramid[l].at<cv::Vec3b>(i, j)[2] = 128 - 5;
						}
						else {
							B_prime_pyramid[l].at<cv::Vec3b>(i, j)[1] = B_pyramid[l].at<cv::Vec3b>(i, j)[1];
							B_prime_pyramid[l].at<cv::Vec3b>(i, j)[2] = B_pyramid[l].at<cv::Vec3b>(i, j)[2];
						}
						// if()
					}
				}
			}
			for(int i = height_prime-1; i >= 0; i--) {
				for(int j = width_prime-1; j >= 0; j--) {
					if(get_component(B_colored_pyramid[l].at<cv::Vec3b>(i - 7, j)) == 1) {
						B_prime_pyramid[l].at<cv::Vec3b>(i,j) = B_prime_pyramid[l].at<cv::Vec3b>(i - 1,j);
						if(get_component(B_colored_pyramid[l].at<cv::Vec3b>(i, j)) != 1)
							B_prime_pyramid[l].at<cv::Vec3b>(i,j)[0] -= 12;
					}
					if(get_component(B_colored_pyramid[l].at<cv::Vec3b>(i - 5, j)) == 7) {
						B_prime_pyramid[l].at<cv::Vec3b>(i,j) = B_prime_pyramid[l].at<cv::Vec3b>(i - 5,j);
						// B_prime_pyramid[l].at<cv::Vec3b>(i,j)[0] -= 10;
					}
					if(get_component(B_colored_pyramid[l].at<cv::Vec3b>(i - 20, j)) == 9 && get_component(B_colored_pyramid[l].at<cv::Vec3b>(i, j)) == 11) {
						B_prime_pyramid[l].at<cv::Vec3b>(i,j) = B_prime_pyramid[l].at<cv::Vec3b>(i - 1,j);
						// B_prime_pyramid[l].at<cv::Vec3b>(i,j)[0] -= 10;
					}
					if(get_component(B_colored_pyramid[l].at<cv::Vec3b>(i - 5, j)) == 2) {
						B_prime_pyramid[l].at<cv::Vec3b>(i,j) = B_prime_pyramid[l].at<cv::Vec3b>(i - 1,j);
						if(get_component(B_colored_pyramid[l].at<cv::Vec3b>(i, j)) == 3){
							B_prime_pyramid[l].at<cv::Vec3b>(i,j)[0] = B_prime_pyramid[l].at<cv::Vec3b>(i,j)[0]/2 + B_pyramid[l].at<cv::Vec3b>(i,j)[0]/2;
						}
						else if(get_component(B_colored_pyramid[l].at<cv::Vec3b>(i, j)) != 2)
							B_prime_pyramid[l].at<cv::Vec3b>(i,j)[0] -= 10;
					}
					if(get_component(B_colored_pyramid[l].at<cv::Vec3b>(i - 4, j)) == 8 && get_component(B_colored_pyramid[l].at<cv::Vec3b>(i, j)) != 8) {
						B_prime_pyramid[l].at<cv::Vec3b>(i,j) = B_prime_pyramid[l].at<cv::Vec3b>(i - 1,j);
						// B_prime_pyramid[l].at<cv::Vec3b>(i,j)[0] -= 10;
					}
				}
			}
			for(int i = 0; i < height_prime; i++) {
				for(int j = 0; j < width_prime; j++) {
					if(get_component(B_colored_pyramid[l].at<cv::Vec3b>(i + 6, j)) == 0 && get_component(B_colored_pyramid[l].at<cv::Vec3b>(i, j)) == 11) {
						B_prime_pyramid[l].at<cv::Vec3b>(i,j) = B_prime_pyramid[l].at<cv::Vec3b>(i + 1,j);
						// B_prime_pyramid[l].at<cv::Vec3b>(i,j)[0] -= 10;
					}
					if(get_component(B_colored_pyramid[l].at<cv::Vec3b>(i + 5, j)) == 9 && get_component(B_colored_pyramid[l].at<cv::Vec3b>(i, j)) == 6) {
						B_prime_pyramid[l].at<cv::Vec3b>(i,j) = B_prime_pyramid[l].at<cv::Vec3b>(i + 1,j);
						// B_prime_pyramid[l].at<cv::Vec3b>(i,j)[0] -= 10;
					}
					// if(get_component(B_colored_pyramid[l].at<cv::Vec3b>(i + 1, j)) != 3 && get_component(B_colored_pyramid[l].at<cv::Vec3b>(i, j)) == 11) {
					// 	B_prime_pyramid[l].at<cv::Vec3b>(i,j) = B_prime_pyramid[l].at<cv::Vec3b>(i + 1,j);
					// 	// B_prime_pyramid[l].at<cv::Vec3b>(i,j)[0] -= 10;
					// }
				}
			}
		}
	}
	else {
		int color_pixel;
		for(int l = Levels; l >= 0; l--) {
			int height_prime = B_prime_pyramid[l].rows;
			int width_prime = B_prime_pyramid[l].cols;
			int best_i, best_j;
			for(int i = 0; i < height_prime; i++) {
				for(int j = 0; j < width_prime; j++) {
					cout<<"Going for best match "<< i <<" "<<j<<endl;
					color_pixel = best_match(A_pyramid, A_prime_pyramid, B_pyramid, B_prime_pyramid, s_pyramid, A_features, B_features, l, Levels, i, j, best_i, best_j, A_colored_points, B_colored_pyramid);
					s_pyramid[l].at<cv::Vec3b>(i,j)[0] = best_i;
					s_pyramid[l].at<cv::Vec3b>(i,j)[1] = best_j;
					s_pyramid[l].at<cv::Vec3b>(i,j)[2] = 1;
					B_prime_pyramid[l].at<cv::Vec3b>(i,j)[0] = A_prime_pyramid[l].at<cv::Vec3b>(best_i,best_j)[0];
					B_prime_pyramid[l].at<cv::Vec3b>(i,j)[1] = B_pyramid[l].at<cv::Vec3b>(i,j)[1];
					B_prime_pyramid[l].at<cv::Vec3b>(i,j)[2] = B_pyramid[l].at<cv::Vec3b>(i,j)[2];
				}
			}
		}
	}

	cvtColor(B_prime,B_prime,CV_YUV2BGR);
	return B_prime;
}

int main(int argc, char** argv){
	float resize_coefficient = 2;
	// Input images A, A' and B
	Mat A, A_prime, B, B_prime, A_colored, B_colored;
	flag_color = stoi(argv[1]); // 1 indicates aging, 0 indicates simple analogy
	if(flag_color == 1) {
	    A = imread(argv[2], CV_LOAD_IMAGE_COLOR);
	    A_prime = imread(argv[3], CV_LOAD_IMAGE_COLOR);
	    A_colored = imread(argv[4], CV_LOAD_IMAGE_COLOR);
	    B = imread(argv[5], CV_LOAD_IMAGE_COLOR);
	    B_colored = imread(argv[6], CV_LOAD_IMAGE_COLOR);
	    resize(A, A, Size(A.cols/resize_coefficient, A.rows/resize_coefficient));
    	resize(A_prime, A_prime, Size(A_prime.cols/resize_coefficient, A_prime.rows/resize_coefficient));
    	resize(A_colored, A_colored, Size(A_colored.cols/resize_coefficient, A_colored.rows/resize_coefficient));
    	resize(B, B, Size(B.cols/resize_coefficient, B.rows/resize_coefficient));
    	resize(B_colored, B_colored, Size(B_colored.cols/resize_coefficient, B_colored.rows/resize_coefficient));
	}
	else {
		A = imread(argv[2], CV_LOAD_IMAGE_COLOR);
	    A_prime = imread(argv[3], CV_LOAD_IMAGE_COLOR);
	    B = imread(argv[4], CV_LOAD_IMAGE_COLOR);
		resize(A, A, Size(A.cols/resize_coefficient, A.rows/resize_coefficient));
    	resize(A_prime, A_prime, Size(A_prime.cols/resize_coefficient, A_prime.rows/resize_coefficient));
    	resize(B, B, Size(B.cols/resize_coefficient, B.rows/resize_coefficient));
    	Mat tempA(A.rows, A.cols, CV_8UC3, Scalar(0,0,0));
    	Mat tempB(B.rows, B.cols, CV_8UC3, Scalar(0,0,0));
    	A_colored = tempA; // Set as dummy 
    	B_colored = tempB; // Set as dummy
	}

	view_image(A, "A");
	view_image(A_prime, "A_prime");
	wait_view_image(B, "B");
    B_prime = create_image_analogy(A, A_prime, A_colored, B, B_colored);
	wait_view_image(B_prime, "B_prime");
	return 0;
}