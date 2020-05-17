#include <iostream>
#include <stdio.h>
#include <ctime>
#include "tbb/parallel_for.h"
#include "tbb/blocked_range2d.h"
#include "tbb/task_scheduler_init.h"
#include <omp.h>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <chrono>

using namespace cv;
using namespace tbb;
using namespace std;

#define M_PI 3.14159265358979323846

int Clamp(int value, int min, int max)
{
	if (value < min)
		return min;

	if (value > max)
		return max;

	return value;
}

void printMatrix(double** matrix, int n, int m) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			cout << setw(4) << setprecision(2) << matrix[i][j] << "   ";
		}
		cout << endl;
	}
}

void deleteMatrix(double** matrix, int n) {
	for (int i = 0; i < n; i++) {
		delete[] matrix[i];
	}
	delete matrix;
}


//-------------------------------------------------------------//
//-------------------------------------------------------------//
//-------------------------------------------------------------//
//-------------------------------------------------------------//
//-------------------------------------------------------------//

double** createKernel(int n, int m) {
	double** mas = new double* [n];

	for (int i = 0; i < n; i++) {
		mas[i] = new double[m];
	}

	return mas;
}
void InitKern(double** kernel, int k, double sigma)
{
	int radius = (k / 2);
	double sum = 0.0;
	double tmp = 1 / (2 * M_PI * sigma * sigma);

	for (int i = -radius; i <= radius; i++)
		for (int j = -radius; j <= radius; j++)
		{
			kernel[i + radius][j + radius] = tmp * exp(-(i * i + j * j) / (2 * sigma * sigma));
			sum += kernel[i + radius][j + radius];
		}
	for (int i = -radius; i <= radius; i++)
		for (int j = -radius; j <= radius; j++)
		{
			kernel[i + radius][j + radius] /= sum;
		}
}

void mat_cols_mult(double** kernel, const Mat& input, Mat output, int i, int radius)
{
	for (size_t j = 0; j < input.cols; j++)
	{
		double tmp = 0;
		for (int l = -radius; l <= radius; l++)
			for (int w = -radius; w <= radius; w++)
			{
				int idX = Clamp(i + l, 0, input.rows - 1);
				int idY = Clamp(j + w, 0, input.cols - 1);
				tmp += double(input.at<uchar>(idX, idY)) * kernel[l + radius][w + radius];
			}
		output.at<uchar>(i, j) = int(tmp);
	}
}

void parallel_matrix_multiply(double** kernel, const Mat& input, Mat& output, int k)
{
	int radius = k / 2;
	parallel_for(blocked_range<size_t>(0, input.rows), [=](const blocked_range<size_t>& r)
		{
			for (size_t i = r.begin(); i != r.end(); ++i)
			{
				mat_cols_mult(kernel, input, output, i, radius);
			}
		});
}


int main() {
	Mat image;
	image = imread("Nature_Other_Stream_from_the_mountains_031199_.jpg", IMREAD_COLOR);
	if (!image.data)
	{
		cout << "No image data" << endl;
		return -1;
	}

	Mat gray_image;
	Mat my_result;
	cvtColor(image, gray_image, COLOR_BGR2GRAY);
	cvtColor(image, my_result, COLOR_BGR2GRAY);

	int k = 3;

	double** Kernel = createKernel(k, k);
	InitKern(Kernel, k, 7);

	auto begin = std::chrono::steady_clock::now();
	task_scheduler_init init(6);
	parallel_matrix_multiply(Kernel, gray_image, my_result, k);
	auto end = std::chrono::steady_clock::now();
	auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
	cout << "The time is: " << elapsed_ms.count() << " ms\n";

	imwrite("../../images/Gray_Image.jpg", gray_image);
	namedWindow("my_result", WINDOW_AUTOSIZE);
	namedWindow("Gray image", WINDOW_AUTOSIZE);
	imshow("my_result", my_result);
	imshow("Gray image", gray_image);
	waitKey(0);
	deleteMatrix(Kernel, k);
	return 0;
}