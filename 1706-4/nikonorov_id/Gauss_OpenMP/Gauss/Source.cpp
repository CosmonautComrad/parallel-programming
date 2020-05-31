#include <iostream>
#include <ctime>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <omp.h>
#include <chrono>

using namespace cv;
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

	delete [] kernel;
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

void OpenMP_Gauss(const Mat& input, Mat& output, double* kernel, int kern_size, int thread_nums)
{
	int radius = int(kern_size / 2);
#pragma omp parallel for num_threads(thread_nums)
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
		return false;

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
	Mat OpenMP_result;
	cvtColor(original_image, gray_image, COLOR_BGR2GRAY);
	cvtColor(original_image, seq_result, COLOR_BGR2GRAY);
	cvtColor(original_image, OpenMP_result, COLOR_BGR2GRAY);

	int k = 3;

	double* kernel;
	kernel = createKernel(k, k);
	InitKern(kernel, k, 7);

	auto begin = std::chrono::steady_clock::now();
	OpenMP_Gauss(gray_image, seq_result, kernel, k, thread_nums);
	auto end = std::chrono::steady_clock::now();
	auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
	cout << "The OpenMP time is: " << elapsed_ms.count() << " ms" << endl;
	
	auto s_begin = std::chrono::steady_clock::now();
	Sequential_Gauss(gray_image, OpenMP_result, kernel, k);
	auto s_end = std::chrono::steady_clock::now();
	auto s_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(s_end - s_begin);
	cout << "The sequential time is: " << s_elapsed_ms.count() << " ms" << endl;

	if (check(seq_result, OpenMP_result))
	{
		imwrite(result_image_path, OpenMP_result);
		namedWindow("my_result", WINDOW_AUTOSIZE);
		namedWindow("Gray image", WINDOW_AUTOSIZE);
		imshow("my_result", OpenMP_result);
		imshow("Gray image", gray_image);
		waitKey(0);
	}
	deletekernel(kernel, k);
	return 0;
}