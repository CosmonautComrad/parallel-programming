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

void deletekernel(double* kernel, int k) {

	delete[] kernel;
}


//-------------------------------------------------------------//
//-------------------------------------------------------------//
//-------------------------------------------------------------//
//-------------------------------------------------------------//
//-------------------------------------------------------------//

double* createKernel(int rows, int cols) {
	int tmp = rows * cols;
	double* mas;
	mas = new double[tmp];
	return mas;
}
void InitKern(double* kernel, int k, double sigma)
{
	int radius = (k / 2);
	double sum = 0.0;
	double tmp = 1 / (2 * M_PI * sigma * sigma);
	int step = radius;

	for (int i = -radius; i <= radius; i++)
	{
		for (int j = -radius; j <= radius; j++)
		{
			kernel[j + step] = tmp * exp(-(i * i + j * j) / (2 * sigma * sigma));
			sum += kernel[j + step];
		}
		step += k;
	}

	for (int i = 0; i < k * k; i++)
	{
		kernel[i] /= sum;
	}
}

void mat_cols_mult(double* kernel, const Mat& input, Mat output, int i, int radius)
{
	for (size_t j = 1; j < input.cols; j++)
	{
		double tmp = 0;
		for (int l = -radius; l <= radius; l++)
			for (int w = -radius; w <= radius; w++)
			{
				int idX = Clamp(i + l, 0, input.rows - 1);
				int idY = Clamp(j + w, 0, input.cols - 1);
				tmp += double(input.at<uchar>(idX, idY)) * kernel[l + w + 2 * (l + 2)];
			}
		output.at<uchar>(i, j) = int(tmp);
	}
}


void parallel_matrix_multiply(double* kernel, const Mat& input, Mat& output, int k)
{
	int radius = k / 2;
	parallel_for(blocked_range<size_t>(1, input.rows), [=](const blocked_range<size_t>& r)
		{
			for (size_t i = r.begin(); i != r.end(); ++i)
			{
				mat_cols_mult(kernel, input, output, i, radius);
			}
		});
}

void Sequential_Gauss(const Mat& input, Mat& output, double* kernel, int kern_size)
{
	int radius = int(kern_size / 2);
	for (int i = 1; i < input.rows; i++) {
		for (int j = 1; j < input.cols; j++) {
			double tmp = 0.0;
			for (int l = -radius; l <= radius; l++) {
				for (int w = -radius; w <= radius; w++) {
					int idX = Clamp(i + l, 0, input.rows - 1);
					int idY = Clamp(j + w, 0, input.cols - 1);
					tmp += double(input.at<uchar>(idX, idY)) * kernel[l + w + 2 * (l + 2)];
				}
			}
			output.at<uchar>(i, j) = int(tmp);
		}
	}
}

bool check(const Mat& image1, const Mat& image2)
{
	Mat res;
	bitwise_xor(image1, image2, res);
	if (countNonZero(res) > 0)
	{
		cout << res;
		return false;
	}
	else
		return true;
}

int main(int argc, char** argv) {

	string orig_image_path;
	string result_image_path;
	int thread_nums;

	if (argc < 4)
	{
		orig_image_path += "pic.jpg";
		result_image_path += "result.jpg";
		thread_nums = 6;
	}
	else
	{
		orig_image_path += argv[1];
		result_image_path += argv[2];
	}

	Mat original_image;
	original_image = imread(orig_image_path, IMREAD_COLOR);

	if (original_image.empty())
	{
		cout << "No image data" << endl;
		return -1;
	}

	Mat gray_image;
	Mat seq_result;
	Mat TBB_result;
	cvtColor(original_image, gray_image, COLOR_BGR2GRAY);
	cvtColor(original_image, seq_result, COLOR_BGR2GRAY);
	cvtColor(original_image, TBB_result, COLOR_BGR2GRAY);

	int k = 3;

	double* kernel;
	kernel = createKernel(k, k);
	InitKern(kernel, k, 7);

	auto begin = std::chrono::steady_clock::now();
	task_scheduler_init init(thread_nums);
	parallel_matrix_multiply(kernel, gray_image, TBB_result, k);
	auto end = std::chrono::steady_clock::now();
	auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
	cout << "The TBB time is: " << elapsed_ms.count() << " ms\n";

	auto s_begin = std::chrono::steady_clock::now();
	Sequential_Gauss(gray_image, seq_result, kernel, k);
	auto s_end = std::chrono::steady_clock::now();
	auto s_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(s_end - s_begin);
	cout << "The sequential time is: " << s_elapsed_ms.count() << " ms" << endl;

	if (check(seq_result, TBB_result))
	{
		imwrite(result_image_path, TBB_result);
		namedWindow("my_result", WINDOW_AUTOSIZE);
		namedWindow("Gray image", WINDOW_AUTOSIZE);
		imshow("my_result", TBB_result);
		imshow("Gray image", gray_image);
		waitKey(0);
	}

	deletekernel(kernel, k);

	return 0;
}