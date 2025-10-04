#include <opencv2/opencv.hpp>
#include <cmath>

int camWidth = 640;
int camHeight = 480;
int dispWidth = 1080;
int dispHeight = 720;

int main() {

    cv::VideoCapture cap(0); 

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open webcam." << std::endl;
        return -1;
    }

    // set webcam resolution
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  camWidth);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, camHeight);

    // int camWidth = 100;
    // int camHeight = 100;
    int depthMax = 254; // min value must be 1, so z range is 0..254
 

    cv::namedWindow("hill", cv::WINDOW_AUTOSIZE);

    //           rows   cols 
    cv::Mat hill(camHeight, camWidth, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::Mat frame;

    // grab frame to get type
    cap >> frame;
    if(frame.empty()){
        std::cerr<< "Capture failed" << std::endl;
        exit(0);
    }

    cv::Mat renderBuf = frame.clone();

    int mtype = frame.type();        

    int pixelHistory;
    int bufStart = 0;

    int bufSize = 0;

    int framenum = 0;

    //dummy run to get buffer size
    for(int x = 0; x < camWidth; x++){

        for(int y = 0; y < camHeight; y++){

            bufSize += 1 + int(depthMax * cos(M_PI * x /camWidth - M_PI/2) * cos(M_PI * y/camHeight - M_PI/2)); 
        }
    }

    cv::Mat lineBuffer(bufSize + depthMax + depthMax, 1, mtype);

    std::cout << "(" << camWidth << "," << camHeight << ") = " << "\t buffer size: " << bufSize << std::endl;

    while (true) {

        bufStart = 0;

        // grab webcam frame
        cap >> frame; 
        if(frame.empty()){
            std::cerr<< "Capture failed" << std::endl;
            exit(0);
        }

        // 
        for(int x = 0; x < camWidth; x++){
        
            for(int y = 0; y < camHeight; y++){

                int z = 1 + int(depthMax * cos(M_PI * x /camWidth - M_PI/2) * cos(M_PI * y/camHeight - M_PI/2)); 

                // stash this frame's pixel
                lineBuffer.at<cv::Vec3b>(bufStart + (framenum % z), 0) = frame.at<cv::Vec3b>(y,x);

                // pull older pixel
                renderBuf.at<cv::Vec3b>(y,x) = lineBuffer.at<cv::Vec3b>(bufStart + ((framenum + 1)% z), 0);

                bufStart += z;

                //std::cout << "(" << x << "," << y << ") = " << z << "\t sum to date: " << bufStart << std::endl;
            }
        }

        cv::imshow("hill", renderBuf);

        framenum++;


        // Exit loop if 'q' is pressed
        if (cv::waitKey(1) == 'q') {
            break;
        }
    

       
    }// true

    return 0;
}

