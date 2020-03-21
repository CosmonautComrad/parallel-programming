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
void Gauss(Mat input, Mat output, double** kernel, int kern_size)
{
	int radius = int(kern_size / 2);
	for (int i = 1; i < input.rows; i++) {
		for (int j = 1; j < input.cols; j++) {
			double tmp = 0.0;
			for (int l = -radius; l <= radius; l++) {
				for (int w = -radius; w <= radius; w++) {
					int idX = Clamp(i + l, 0, input.rows - 1);
					int idY = Clamp(j + w, 0, input.cols - 1);
					tmp += double(input.at<uchar>(idX, idY)) * kernel[l + radius][w + radius];
				}
			}
			output.at<uchar>(i, j) = int(tmp);

		}
	}
}

int main() {
	Mat image;
	image = imread("anime.jfif", IMREAD_COLOR);
	if (!image.data)
	{
		printf(" No image data \n ");
		return -1;
	}
	Mat gray_image;
	Mat my_result;
	cvtColor(image, gray_image, COLOR_BGR2GRAY);
	cvtColor(image, my_result, COLOR_BGR2GRAY);

	int k = 3;

	double** Kernel = createKernel(k, k);
	auto begin = std::chrono::steady_clock::now();
	InitKern(Kernel, k, 7);
	Gauss(gray_image, my_result, Kernel, k);
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