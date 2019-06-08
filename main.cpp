#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include<opencv2/core/core.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void drawAndShowContours(Size imageSize, vector<vector<Point>> contours, string windowName)
{
	Mat result(imageSize, CV_8UC3, Scalar(0, 0, 0));
	drawContours(result, contours, -1, Scalar(255, 255, 255), 1);
	namedWindow(windowName, WINDOW_NORMAL);
	imshow(windowName, result);
}

int main(void)
{
	VideoCapture origVideo;
	Mat frame1;
	Mat	frame2;

	origVideo.open("MP1920x1080.mp4");	//Reference video
	if (!origVideo.isOpened())
	{
		cout << endl << "Error reading video file" << endl;
		waitKey(0);
		return(0);
	}

	char checkEsc = 0;
	bool firstFrame = true;
	int frameCount = 2;

	//	The original location of the center of the tracking ellipse.
	//	Currently calculated manually.
	int centerPreviousEllipseX = 875;
	int centerPreviousEllipseY = 617;

	origVideo.read(frame1);
	origVideo.read(frame2);

	while (origVideo.isOpened() && checkEsc != 27)
	{
//		waitKey(50);	//	Low-Res reference video needs delay

		//	Uncomment to see this step
//		namedWindow("Original Video", WINDOW_NORMAL);
//		imshow("Original Video", frame1);

		//	Frames, that
		Mat frame1Copy = frame1.clone();
		Mat	frame2Copy = frame2.clone();

		Mat pyrFrameForProccess1;	//	First frame for processing
		Mat pyrFrameForProccess2;	//	Second frame for processing

		Mat frameMarker = frame1.clone();	//	Frames for result video

		Mat binaryCheck = frame1.clone();	//	Frames for check the presence of the blade

		//	Down-scale and upscale the image to filter out the noise.
		//	It also speeds up the processing
		float resizeCoefficientX = frame1Copy.cols / 320;	//	Resize coefficient X
		float resizeCoefficientY = frame1Copy.rows / 200;	//	Resize coefficient Y
		resize(frame1Copy, pyrFrameForProccess1, Size(320, 200));
		resize(frame2Copy, pyrFrameForProccess2, Size(320, 200));

		//	Blurring frames
		GaussianBlur(pyrFrameForProccess1, pyrFrameForProccess1, Size(11, 11), 0);
		GaussianBlur(pyrFrameForProccess2, pyrFrameForProccess2, Size(11, 11), 0);
		GaussianBlur(binaryCheck, binaryCheck, Size(11, 11), 0);
		//	Uncomment to see this step
//		namedWindow("GaussianBlur", WINDOW_NORMAL);
//		imshow("GaussianBlur", pyrFrameForProccess1);

		//	Converting frames to grayscale
		cvtColor(pyrFrameForProccess1, pyrFrameForProccess1, COLOR_BGR2GRAY);
		cvtColor(pyrFrameForProccess2, pyrFrameForProccess2, COLOR_BGR2GRAY);
		cvtColor(binaryCheck, binaryCheck, COLOR_BGR2GRAY);
		//	Uncomment to see this step
//		namedWindow("Gray", WINDOW_NORMAL);
//		imshow("Gray", pyrFrameForProccess1);
		
		threshold(binaryCheck, binaryCheck, 60, 255, THRESH_BINARY_INV);
		//	Uncomment to see this step
//		namedWindow("Binary Check", WINDOW_NORMAL);
//		imshow("Binary Check", binaryCheck);

		Mat frameDifferences;
		//	Calculated absolute difference between two images
		absdiff(pyrFrameForProccess1, pyrFrameForProccess2, frameDifferences);
		//	Uncomment to see this step
//		namedWindow("Differens between frames", WINDOW_NORMAL);
//		imshow("Differens between frames", frameDifferences);

		Mat frameBinary;
		threshold(frameDifferences, frameBinary, 35, 255, THRESH_BINARY);
		namedWindow("Binary", WINDOW_NORMAL);
		imshow("Binary", frameBinary);

		//Morphology operations
		Mat structuringElement5x5 = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
		Mat structuringElement9x9 = getStructuringElement(MORPH_RECT, Size(9, 9));
		Mat structuringElement13x13 = getStructuringElement(MORPH_ELLIPSE, Size(13, 13));
		Mat structuringElement15x15 = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));
		Mat structuringElement19x19 = getStructuringElement(MORPH_ELLIPSE, Size(19, 19));
		Mat structuringElement21x21 = getStructuringElement(MORPH_ELLIPSE, Size(21, 21));

		Mat morphBinary = frameBinary.clone();
		for (int i = 0; i < 4; i++)
		{
			dilate(morphBinary, morphBinary, structuringElement21x21);
		}
		//	Uncomment to see this step
//		namedWindow("Morphology", WINDOW_NORMAL);
//		imshow("Morphology", morphBinary);

		vector<vector<Point>> contours;
		findContours(morphBinary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		drawAndShowContours(morphBinary.size(), contours, "Contours");

		vector<vector<Point>> convexHulls(contours.size());
		for (unsigned int i = 0; i < contours.size(); i++) {
			convexHull(contours[i], convexHulls[i]);
		}

		//	Blade elipse (area) search
		vector<RotatedRect> minEllipse(convexHulls.size());
		for (size_t i = 0; i < convexHulls.size(); i++)
		{
			if (convexHulls[i].size() > 5)
			{
				minEllipse[i] = fitEllipse(convexHulls[i]);
			}
		}

		//	Resize ellipse.
		//	This is done to put an ellipse on the reference video size
		minEllipse[0].center.x = minEllipse[0].center.x * resizeCoefficientX;
		minEllipse[0].center.y = minEllipse[0].center.y * resizeCoefficientX;
		minEllipse[0].size.width = minEllipse[0].size.width * (resizeCoefficientX - 1);
		minEllipse[0].size.height = minEllipse[0].size.height * (resizeCoefficientX - 1);

		//	Check ellipse offset.
		//	This is to prevent interference.
		if ((abs(minEllipse[0].center.x - centerPreviousEllipseX) > 10) ||
			(abs(minEllipse[0].center.y - centerPreviousEllipseY) > 10))
		{
			minEllipse[0].center.x = centerPreviousEllipseX;
			minEllipse[0].center.y = centerPreviousEllipseY;
		}
		centerPreviousEllipseX = minEllipse[0].center.x;
		centerPreviousEllipseY = minEllipse[0].center.y;

		//	Draw an ellipse
		for (size_t i = 0; i < minEllipse.size(); i++)
		{
			ellipse(frameMarker, minEllipse[i], Scalar(0, 255, 0), 5);
		}
		//	Uncomment to see this step
//		namedWindow("Ellipse", WINDOW_NORMAL);
//		imshow("Ellipse", frameMarker);

		//	The pixel in which the tracking of the blade presence (0 or 1) occurs
		int trackedPixel = (int)binaryCheck.at<unsigned char>
			(minEllipse[0].center.y - (minEllipse[0].size.height / 5),
			minEllipse[0].center.x);

		//	Timer
		float time = frameCount * 0.03;
		cout << endl << "Time: " << time << endl;
		cout << "Pixel value: " << trackedPixel << endl;

		drawMarker(frameMarker, Point(minEllipse[0].center.x,
										minEllipse[0].center.y - (minEllipse[0].size.height / 5)),
										Scalar(0, 0, 255));
		namedWindow("Result", WINDOW_NORMAL);
		imshow("Result", frameMarker);

		frame1 = frame2.clone();
		if ((origVideo.get(CAP_PROP_POS_FRAMES) + 1) < origVideo.get(CAP_PROP_FRAME_COUNT))
		{
			origVideo.read(frame2);
		}
		else
		{
			cout << "End of video" << endl;
			break;
		}

		firstFrame = false;
		frameCount++;
		checkEsc = waitKey(1);
	}

	if (checkEsc != 27)
	{
		waitKey(0);
	}
	return(0);
}

/* End of file */