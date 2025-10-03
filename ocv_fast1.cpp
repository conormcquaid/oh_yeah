#include <opencv2/opencv.hpp>

int main() {
    // 1. Open the default camera (index 0)
    cv::VideoCapture cap(0); 

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open webcam." << std::endl;
        return -1;
    }

    cv::Mat frame;
    cv::namedWindow("Processed Frame", cv::WINDOW_AUTOSIZE);

    while (true) {
        // Read a new frame from the camera
        cap >> frame; 

        if (frame.empty()) {
            std::cerr << "Error: Frame is empty." << std::endl;
            break;
        }

        // --- 2. Transformation ---
        // Example transformation: Convert to grayscale and blur
        cv::Mat processed_frame;
        cv::cvtColor(frame, processed_frame, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(processed_frame, processed_frame, cv::Size(5, 5), 0);

        // --- 3. Render ---
        cv::imshow("Processed Frame", processed_frame);

        // Exit loop if 'q' is pressed
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    return 0;
}

