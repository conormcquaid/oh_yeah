#include <opencv2/opencv.hpp>
#include <cmath>
#include <unistd.h>

//// To find out capabilities of the camera...
//// v4l2-ctl --list-formats-ext



/*
    The following EffOfEcksWye could be parameterized
    int(depthMax * cos(M_PI * x /camWidth - M_PI/2) * cos(M_PI * y/camHeight - M_PI/2)); 

*/

int camWidth = 1080;
int camHeight = 1080;
int dispWidth = 1920;
int dispHeight = 1080;
int depthMax = 64; // min value must be 1, so depthMax cannot exceed 254
bool verbose = false;
bool debugWindow = false;
int filterOption = 0;
char* progname;
int fps;
bool mirrorHorizontal = false;
bool mirrorVertical = false;

bool assertInRange(int n, int min, int max);
void load_options(int argc, char** argv);
void usage(void);
 

int main(int argc, char** argv) {

    progname = argv[0];

    load_options(argc, argv);

    // TODO: a means of counting the available cameras
    // and an option for choosing one
    cv::VideoCapture cap(0); 

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open webcam." << std::endl;
        return -1;
    }

    // set webcam resolution
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J','P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  camWidth);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, camHeight); 
    cap.set(cv::CAP_PROP_FPS, 30);

    int setfps = 60;
    while(fps != setfps && setfps > 10){
        cap.set(cv::CAP_PROP_FPS, setfps);
        fps = cap.get(cv::CAP_PROP_FPS);
        setfps--;
    }

    cv::namedWindow("hill", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("debug", cv::WINDOW_AUTOSIZE);

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

    //TODO: grab _actual_ camera dimensions
    camWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    camHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    fps = cap.get(cv::CAP_PROP_FPS);

    if(verbose) std::cout << "Maximum history per pixel is " << depthMax <<  std::endl;
    if(verbose) std::cout << "Camera WxH ( " << camWidth << ", " << camHeight << " ) = " << "\t buffer size: " << bufSize << std::endl;
    if(verbose) std::cout << "Screen WxH ( " << renderBuf.cols << ", " << renderBuf.rows << " )" <<  std::endl;
    if(verbose) std::cout << "Frames per second: " << fps <<  std::endl;


    while (true) {

        bufStart = 0;

        // grab webcam frame
        cap >> frame; 
        if(frame.empty()){
            std::cerr<< "Capture failed" << std::endl;
            exit(0);
        }

        if(verbose)std::cout << framenum << std::endl;
 
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

        cv::Size outputSize( dispHeight, dispWidth);
        cv::resize(final, renderBuf, outputSize);

        //TODO: apply filtering here? or before resizing?

        cv::imshow("hill", final);

        if(debugWindow){
            cv::imshow("debug", frame);
        }

        framenum++;


        // Exit loop if 'q' is pressed
        if (cv::waitKey(1) == 'q') {
            break;
        }
    

       
    }// true

    return 0;
}


//////////////////////////////
void usage(void){
    std::cout << "Usage:" << progname << " [OPTIONS]" << std::endl;
    std::cout << "2D funhouse mirror effect" << std::endl << std::endl;
    std::cout << "Hit 'Q' to quit" << std::endl << std::endl;
    std::cout << "-v\tVerbose" << std::endl;
    std::cout << "-g\tShow debug window (raw camera feed)" << std::endl;
    std::cout << "-d\tSet maximum pixel history depth" << std::endl;
    std::cout << "-w\tSet camera width" << std::endl;
    std::cout << "-h\tSet camera heigth" << std::endl;
    std::cout << "-W\tSet output window width" << std::endl;
    std::cout << "-H\tSet output window height" << std::endl;
    std::cout << "-f\tSet filter to apply to output" << std::endl;
    std::cout << "-m\tMirror horizontally" << std::endl;
    std::cout << "-M\tMirror vertically" << std::endl;
    std::cout << "v4l2-ctl --list-formats-ext will show devices and capabilities" << std::endl;
    std::cout << std::endl;
}

bool assertInRange(int n, int min, int max){

    if( n >= min && n <= max){
        return true;
    }else{
        usage();
        exit(0);
    }
    return false;// happy now, gcc?
}

void load_options(int argc, char** argv){
    int opt;

    if(argc < 2){
        std::cout << "(just FYI)" << std::endl;
        usage();
    }

    while((opt = getopt(argc, argv, "gvd:w:W:h:H:f:")) != -1){
        switch(opt){
            case 'v':
            verbose = true;
            break;
            
            case 'd':
            depthMax = atoi(optarg);
            assertInRange(depthMax, 1, 254);
                
            break;

            case 'h':
            camHeight = atoi(optarg);
            assertInRange(camHeight, 1, 2048);
            break;

            case 'H':
            dispHeight = atoi(optarg);
            assertInRange(dispHeight, 1, 2048);
            break;

            case 'w':
            camWidth = atoi(optarg);
            assertInRange(camWidth, 1, 2048);
            break;

            case 'W':
            dispWidth = atoi(optarg);
            assertInRange(dispWidth, 1, 2048);
            break;

            case 'f':
            filterOption = atoi(optarg);
            assertInRange(filterOption, 0, 999);
            break;

            case 'g':
            debugWindow = true;
            break;

            case '?':
            std::cout << "I'm don't know what you mean by '" << opt <<"'" << std::endl;
            break;

            
        }

    }
}

