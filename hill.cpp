#include <opencv2/opencv.hpp>
#include <cmath>
#include <unistd.h>
#include <chrono>
#include <iomanip>
//// To find out capabilities of the camera...
//// v4l2-ctl --list-formats-ext
//// To compile with profiling enabled...
//// cmake -DCMAKE_CXX_FLAGS=-pg -DCMAKE_EXE_LINKER_FLAGS=-pg -DCMAKE_SHARED_LINKER_FLAGS=-pg .


/*
    The following EffOfEcksWye could be parameterized
    int(depthMax * cos(M_PI * x /camWidth - M_PI/2) * cos(M_PI * y/camHeight - M_PI/2)); 

*/

int camWidth   = 1280;
int camHeight  = 720;
int dispWidth  = 1920;
int dispHeight = 1080;
int depthMax = 64; // min value must be 1, so depthMax cannot exceed 254
bool verbose = false;
bool debugWindow = false;
int filterOption = 0;
char* progname;
int fps;
bool mirrorHorizontal = false;
bool mirrorVertical = false;
int param = 0;



bool assertInRange(int n, int min, int max);
void load_options(int argc, char** argv);
void usage(void);

// we can choose diffferent algortithms (if we can think them up ☺ )
#define N_ALGORITHMS 6
int algorithm = 1;
int algo_cosX_x_cosY(int r, int c, int param);
int algo_Inv_cosX_x_cosY(int r, int c, int param);
int algo_OG_param(int r, int c, int param);
int algo_bottom_up(int r, int c, int param);;
int algo_left_to_right(int r, int c, int param);
int algo_right_to_left(int r, int c, int param);

typedef int (*EffOfEcksWye)(int x, int y, int param);
EffOfEcksWye pEffOfEcksWye = algo_cosX_x_cosY;

EffOfEcksWye algorithms[N_ALGORITHMS]={
    algo_cosX_x_cosY,
    algo_Inv_cosX_x_cosY,
    algo_OG_param,
    algo_bottom_up,
    algo_left_to_right,
    algo_right_to_left
};


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

    // in the absence of cam information, try and set fps to the highest possible.
    // this will be dependent on what the camera and the chosen backend allow
    // and non-trivial to automate
    int setfps = 60;
    while(fps != setfps && setfps > 10){
        cap.set(cv::CAP_PROP_FPS, setfps);
        fps = cap.get(cv::CAP_PROP_FPS);
        setfps--;
    }

    // our wo output windows
    cv::namedWindow("hill", cv::WINDOW_AUTOSIZE);
    if(debugWindow){
        cv::namedWindow("debug", cv::WINDOW_AUTOSIZE);
    }



    // where the raw cam data goes...
    cv::Mat frame;

    // grab frame to get type
    cap >> frame;
    if(frame.empty()){
        std::cerr<< "Capture failed" << std::endl;
        exit(0);
    }

    // renderBuf will hold the output of our artistic algorithm
    cv::Mat renderBuf = frame.clone();
    

    // a final output buffer may be needed since resize is not an in-place function]
    int mtype = frame.type();       
    cv::Mat final(dispHeight, dispWidth, mtype);

    //           rows   cols 
    //cv::Mat hill(camHeight, camWidth, CV_8UC3, cv::Scalar(0, 0, 0));

    int bufStart = 0;  // the start of the history for a given pixel
    int bufSize = 0;   // how many pixels needed to hold the total  history
    int framenum = 0;  // ++ each capture frame


    std::vector<std::vector<int>> pixDepth(frame.rows, std::vector<int>(frame.cols));


    // get buffer size and memoize depth function
    for(int c = 0; c < frame.cols; c++){

        for(int r = 0; r < frame.rows; r++){

            int d = pEffOfEcksWye(r,c,param);
            bufSize += d; 
            // cache depth calculation
            pixDepth[r][c] = d;

        }
    }

    cv::Mat lineBuffer(bufSize + depthMax + depthMax, 1, mtype);

    //TODO: grab _actual_ camera dimensions
    camWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    camHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    fps = cap.get(cv::CAP_PROP_FPS);

    if(verbose) std::cout << "Maximum history per pixel is " << depthMax <<  std::endl;
    if(verbose) std::cout << "Camera WxH ( " << frame.cols  << ", " << frame.rows  << " )" << std::endl;
    if(verbose) std::cout << "Screen WxH ( " << dispWidth   << ", " << dispHeight  << " ) (requested or default: may not be the actual resulting size)" <<  std::endl;
    if(verbose) std::cout << "History buffer size: " << bufSize << std::endl;
    if(verbose) std::cout << "Frames per second: " << fps <<  std::endl;
    if(verbose && frame.isContinuous()) std::cout << "Optimizing out cv::Mat.at()" <<  std::endl;

    bool once = false;

    // [frame] ==> [lineBuffer] ==> [renderBuf]

    while (true) {

        auto start_frame = std::chrono::high_resolution_clock::now();

        bufStart = 0;

        // grab webcam frame
        cap >> frame; 
        if(frame.empty()){
            std::cerr<< "Capture failed" << std::endl;
            exit(0);
        }

//        if(verbose)std::cout << framenum << "\r" << std::flush;

        if(       frame.isContinuous() && 
                  lineBuffer.isContinuous() &&
                  renderBuf.isContinuous() ){

            cv::Vec3b* pSrc = frame.ptr<cv::Vec3b>(0);
            cv::Vec3b* pBuf = lineBuffer.ptr<cv::Vec3b>(0);
            cv::Vec3b* pOut = renderBuf.ptr<cv::Vec3b>(0);

            for (int c = 0; c < frame.cols; ++c) {
                for (int r = 0; r < frame.rows; ++r) {

                    int z = pixDepth[r][c];
                    
                    int pixelIndex = c + r * frame.cols;

                    pBuf[bufStart + (framenum % z)] = pSrc[pixelIndex];

                    pOut[pixelIndex] = pBuf[bufStart + ((framenum + 1) % z)];
                    
                    bufStart += z;
                }
            }

        }else{
 
            for(int c = 0; c < frame.cols; c++){
            
                for(int r = 0; r < frame.rows; r++){

                    int z = pixDepth[r][c];//pEffOfEcksWye(x,y);
                    // 1 + int(depthMax * cos(M_PI * x /camWidth - M_PI/2) * cos(M_PI * y/camHeight - M_PI/2)); 

                    // stash this frame's pixel
                    lineBuffer.at<cv::Vec3b>(bufStart + (framenum % z), 0) = frame.at<cv::Vec3b>(r,c);

                    // pull older pixel
                    renderBuf.at<cv::Vec3b>(r,c) = lineBuffer.at<cv::Vec3b>(bufStart + ((framenum + 1)% z), 0);

                    bufStart += z;

                    //std::cout << "(" << x << "," << y << ") = " << z << "\t sum to date: " << bufStart << std::endl;
                }
            }
         }

        cv::Size outputSize( dispWidth, dispHeight);
        cv::resize(renderBuf, final, outputSize);

        if(verbose && !once){
            std::cout << "Final WxH ( " << final.cols  << ", " << final.rows  << " )" << std::endl;
        }

        //TODO: apply filtering here? or before resizing?
        int f = depthMax / 2;
        f = (f % 2 == 0) ? f+1 : f;

       //cv::GaussianBlur(final, final, cv::Size(f, f), 0);

        cv::imshow("hill", final);

        if(debugWindow){
            cv::imshow("debug", frame);
        }

        framenum++;


        // Exit loop if 'q' is pressed
        if (cv::waitKey(1) == 'q') {
            break;
        }

        once = true;

        auto end_frame = std::chrono::high_resolution_clock::now();
        if(verbose){
            std::chrono::duration<double, std::milli> frame_duration = end_frame - start_frame;
            std::cout << std::fixed << std::setprecision(0)   << framenum << "\tFPS: " << 1000/frame_duration.count()  << "\r" << std::flush;
        }
    }// true

    return 0;
}

//////////////////////////////

int algo_cosX_x_cosY(int r, int c, int param){

    return 1 + int(depthMax * cos(M_PI * c /camWidth - M_PI/2) * cos(M_PI * r/camHeight - M_PI/2)); 
}

int algo_Inv_cosX_x_cosY(int r, int c, int param){

    return 1 + depthMax - int(depthMax * cos(M_PI * c /camWidth - M_PI/2) * cos(M_PI * r/camHeight - M_PI/2)); 
}

int algo_OG_param(int r, int c, int param){
    if(param < 1) param = 1;
    if(param > depthMax) param = depthMax;
    return (r/param)+1;
}

int algo_bottom_up(int r, int c, int param){
    if(param < 1) param = 1;
    if(param > depthMax) param = depthMax;
    return ((camHeight - r)/param)+1;
}

int algo_left_to_right(int r, int c, int param){
    if(param < 1) param = 1;
    if(param > depthMax) param = depthMax;
    return ((c)/param)+1;
}

int algo_right_to_left(int r, int c, int param){
    if(param < 1) param = 1;
    if(param > depthMax) param = depthMax;
    return ((camWidth - c)/param)+1;
}



//////////////////////////////

void usage(void){
    std::cout << "Usage:" << progname << " [OPTIONS]" << std::endl;
    std::cout << "2D funhouse mirror effect" << std::endl << std::endl;
    std::cout << "Hit 'Q' to quit" << std::endl << std::endl;
    std::cout << "-v\tVerbose" << std::endl;
    std::cout << "-g\tShow debug window (raw camera feed)" << std::endl;
    std::cout << "-d\tSet maximum pixel history depth [1-254]" << std::endl;
    std::cout << "-w\tSet camera width" << std::endl;
    std::cout << "-h\tSet camera heigth" << std::endl;
    std::cout << "-W\tSet output window width" << std::endl;
    std::cout << "-H\tSet output window height" << std::endl;
    std::cout << "-f\tSet filter to apply to output" << std::endl;
    std::cout << "-m\tMirror horizontally" << std::endl;
    std::cout << "-M\tMirror vertically" << std::endl;
    std::cout << "-a\tSelect algorithm 1 - "<< std::to_string(N_ALGORITHMS) << std::endl;
    std::cout << "-p\tOptional algorithm parameter - " << std::endl;
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

    while((opt = getopt(argc, argv, "a:gvd:w:W:h:H:f:p:")) != -1){
        switch(opt){
            case 'v':
            verbose = true;
            break;
 
            case 'a':
            algorithm = atoi(optarg);
            assertInRange(algorithm, 1, N_ALGORITHMS);
            pEffOfEcksWye = algorithms[algorithm - 1];
            break;

            case 'p':
            param = atoi(optarg);
            assertInRange(param, 1, 999);
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

            default:
            /* ignore unknowns */
            break;            
        }
    }
}

