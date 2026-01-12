#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
// #include <opencv2/imgproc.hpp>
// #include <opencv2/videoio.hpp>

// #include <opencv2/highgui.hpp>
#include <cmath>
#include <unistd.h>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <bitset>
#include <string>
#include <queue>
#include <thread>
#include <mutex>
#include <atomic>
#include <cfloat> // For FLT_MAX
#include "PerlinNoise.hpp"



//// To find out capabilities of the camera...
//// v4l2-ctl --list-formats-ext
//// To compile with profiling enabled...
//// cmake -DCMAKE_CXX_FLAGS=-pg -DCMAKE_EXE_LINKER_FLAGS=-pg -DCMAKE_SHARED_LINKER_FLAGS=-pg .


/*
    The following EffOfEcksWye could be parameterized
    int(depthMax * cos(M_PI * x /camWidth - M_PI/2) * cos(M_PI * y/camHeight - M_PI/2)); 

*/

#define USE_CAPTURE_THREAD  99

#define MAX_FILTERS 9
#define K (1024)
#define M (1024 * K)
#define G (1024 * M)

// A whole bunch of globals

int   camWidth     = 1280;
int   camHeight    = 720;
int   dispWidth    = 1920;
int   dispHeight   = 1080;
int   depthMax     = 64; 
bool  verbose      = false;
bool  debugWindow  = false;
int   filterOption = 0;
char* progname;
int   fps;
bool  mirrorHorizontal = false;
bool  mirrorVertical = false;
int   param = 0;
double   variation = 0.0;
double timeout = 0.0;
double noise   = 0.0;
double auto_noise   = 0.0;
std::string  video_device_name = "/dev/video0";


int  enumerate_cameras(void);
void mouse_callback(int event, int x, int y, int flags, void* userdata);
bool assertInRange(int n, int min, int max);
void load_options(int argc, char** argv);
void usage(void);

// we can choose diffferent algortithms (if we can think them up ☺ )
#define N_ALGORITHMS 18
int algorithm = 1;
int algo_cosX_x_cosY(int r, int c, int depth_param, double variation);
int algo_Inv_cosX_x_cosY(int r, int c, int depth_param, double variation);
int algo_OG_param(int r, int c, int depth_param, double variation);
int algo_bottom_up(int r, int c, int depth_param, double variation);;
int algo_left_to_right(int r, int c, int depth_param, double variation);
int algo_right_to_left(int r, int c, int depth_param, double variation);
int algo_cos_x(int r, int c, int depth_param, double variation);
int algo_cos_y(int r, int c, int depth_param, double variation);
int algo_inv_cos_x(int r, int c, int depth_param, double variation);
int algo_inv_cos_y(int r, int c, int depth_param, double variation);
int algo_perlin(int r, int c, int depth_param, double variation);
int algo_concentric(int r, int c, int depth_param, double variation);
int algo_sine(int r, int c, int depth_param, double variation);
int algo_sine2(int r, int c, int depth_param, double variation);
int algo_sine2_highfreq(int r, int c, int depth_param, double variation);
int algo_spiral(int r, int c, int depth_param, double variation);
int algo_multi(int r, int c, int depth_param, double variation, int mode);

typedef int (*EffOfEcksWye)(int x, int y, int depth_param, double variation);
EffOfEcksWye pEffOfEcksWye = algo_cosX_x_cosY;

EffOfEcksWye algorithms[N_ALGORITHMS]={
    algo_cosX_x_cosY,
    algo_Inv_cosX_x_cosY,
    algo_OG_param,
    algo_bottom_up,
    algo_left_to_right,
    algo_right_to_left,
    algo_cos_x,
    algo_cos_y,
    algo_inv_cos_x,
    algo_inv_cos_y,
    algo_perlin,
    algo_perlin,
    algo_perlin,
    algo_concentric,
    algo_sine,
    algo_sine2,
    algo_sine2_highfreq,
    algo_spiral
};

#ifdef USE_CAPTURE_THREAD   
// video capture thread
std::atomic_bool stop_flag(false);
std::queue<cv::Mat> frame_queue;
std::mutex mtx;

void captureTask(cv::VideoCapture& cap){
    cv::Mat frame;
    while(!stop_flag.load() && cap.isOpened()){
        cap >> frame;
        if(frame.empty()){ break; }
        std::lock_guard<std::mutex> lock(mtx);

        // queue can grow arbitrarily large if rendering is slower than capture
        if(frame_queue.size() < 5){
            frame_queue.push(frame.clone());
        }
    }
}
#endif


int main(int argc, char** argv) {

    progname = argv[0]; // remember for usage()

    // inhale command line options
    load_options(argc, argv);

    // TODO: a means of counting the available cameras
    // and an option for choosing one
    // canonoical usage is 'cap(n)' but this often failes to find a usb camera
    // also observed that the first camera may not be /dev/video0
    int cams = enumerate_cameras();
    if(cams == 0){
        std::cerr << "Error: No cameras found" << std::endl;
        return -1;
    }
    if(verbose) std::cout << "Camera bitmask: " << std::bitset<8>(cams) << std::endl;

    cv::VideoCapture cap(video_device_name.c_str());//, cv::CAP_GSTREAMER); <-- gstreamer not built in?

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open webcam." << std::endl;
        return -1;
    }

    // set webcam resolution
    // TODO: be aware that this may not be supported by the camera
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

    // Check for OpenCL/GPU availability
    cv::ocl::setUseOpenCL(true);
    bool haveGpu = cv::ocl::haveOpenCL();
    if(haveGpu){
        if(verbose) std::cout << "OpenCL available: " << cv::ocl::Device::getDefault().name() << std::endl;
    }else{
        if(verbose) std::cout << "OpenCL not available, using CPU" << std::endl;
    }

    // where the raw cam data will go...
    cv::Mat frame;

    // grab one frame just to get type
    cap >> frame;
    if(frame.empty()){
        std::cerr<< "Capture failed" << std::endl;
        exit(0);
    }
    // our two output windows
    cv::namedWindow("slow glass", cv::WINDOW_AUTOSIZE); 
    if(debugWindow){
        cv::namedWindow("debug", cv::WINDOW_AUTOSIZE);
    } 
    cv::Mat thumbnail(frame.rows, frame.cols, CV_8UC1, cv::Scalar(0));

    // renderBuf will hold the output of our artistic algorithm
    cv::Mat renderBuf = frame.clone();
    

    // a final output buffer may be needed since resize is not an in-place function ??
    int mtype = frame.type();
    cv::Mat final(dispHeight, dispWidth, mtype);

    // GPU buffers for accelerated post-processing
    cv::UMat renderGpu, finalGpu;

    //           rows   cols 
    //cv::Mat hill(camHeight, camWidth, CV_8UC3, cv::Scalar(0, 0, 0));


    //attach mouse handler to main window
    //cv::setMouseCallback("hill", mouse_callback, NULL);

    // 
    std::vector<std::vector<int>> pixDepth(frame.rows, std::vector<int>(frame.cols));

#ifdef USE_CAPTURE_THREAD
    // producer thread
    std::thread captureThread(captureTask, std::ref(cap));
#endif

    bool quit = false;
    while(!quit){

        int bufStart = 0;  // the start of the history for a given pixel
        size_t bufSize = 0;   // how many pixels needed to hold the total  history
        int framenum = 0;  // ++ each capture frame

        auto beginTimeout = std::chrono::steady_clock::now();

        // thumbnail shrinks
        cv::resize(thumbnail, thumbnail, cv::Size(frame.cols, frame.rows));
        
        // TODO: variation is only used for perlin noise atm.
        // depth_param is used in updown, left/right, but leaving at 0 for now
        std::srand(std::time(0));
        variation = std::rand() % 1024 / 1024.0;

        // get buffer size and memoize depth function
        int depthMaxEmpirical = 0;
        for(int c = 0; c < frame.cols; c++){

            for(int r = 0; r < frame.rows; r++){

                int d = pEffOfEcksWye(r,c, param, variation);
                depthMaxEmpirical = (d > depthMaxEmpirical) ? d : depthMaxEmpirical;
                if(d < 1) d = 1; // algorithms should already take care of this
                bufSize += d; 
                // cache depth calculation
                pixDepth[r][c] = d;
            }
        }
        // now we can scale thumbnail grayscale
        for(int c = 0; c < frame.cols; c++){
            for(int r = 0; r < frame.rows; r++){
                thumbnail.at<uchar>(r,c) = 255 * pixDepth[r][c] / depthMaxEmpirical;
            }
        }
        // Display the image
        cv::resize(thumbnail, thumbnail, cv::Size(480, 320));
        cv::imwrite("thumbnail.png", thumbnail);
        cv::imshow("Thumbnail", thumbnail);


        cv::Mat lineBuffer(bufSize + depthMax + depthMax, 1, mtype); //pair with lineBuffer.release()

        //TODO: grab _actual_ camera dimensions
        camWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        camHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        fps = cap.get(cv::CAP_PROP_FPS);

        // we can optimize access if all buffers are allocated continuously
        bool continuousBuffers =    frame.isContinuous() && 
                                    lineBuffer.isContinuous() &&
                                    renderBuf.isContinuous();
        if(  continuousBuffers ){

                if(verbose ) std::cout << "Optimizing out cv::Mat.at()" <<  std::endl;
        }
        std::string humanbuf;
        if(bufSize > G){
            humanbuf = " (" + std::to_string(bufSize / G) + " GB)";
        }else if(bufSize > M){
            humanbuf = " (" + std::to_string(bufSize / M) + " MB)"; 
        }else if(bufSize > K){
            humanbuf = " (" + std::to_string(bufSize / K) + " KB)";
        }else{
            humanbuf = " (" + std::to_string(bufSize) + " bytes)";
        }


        if(verbose) std::cout << "Maximum history per pixel is " << depthMax <<  std::endl;
        if(verbose) std::cout << "Camera w x h ( " << frame.cols  << ", " << frame.rows  << " )" << std::endl;
        if(verbose) std::cout << "Screen W x H ( " << dispWidth   << ", " << dispHeight  << " ) (requested or default: may not be the actual resulting size)" <<  std::endl;
        if(verbose) std::cout << "History buffer size: " << bufSize << humanbuf <<std::endl;
        if(verbose) std::cout << "Frames per second: " << fps <<  std::endl;
        if(verbose && timeout) std::cout << "Timeout: " << timeout << " seconds" << std::endl;
        if(verbose) std::cout << "Mirror horizontal: " << (mirrorHorizontal ? "ON" : "OFF") << std::endl;
        if(verbose) std::cout << "Mirror vertical: " << (mirrorVertical ? "ON" : "OFF") << std::endl;
        if(verbose && continuousBuffers) std::cout << "Optimizing out cv::Mat.at()" <<  std::endl <<  std::endl;

        bool once = false;
        bool restart = false;

        // [frame] ==> [lineBuffer] ==> [renderBuf]
        
        cv::Vec3b* pSrc;
        cv::Vec3b* pBuf;
        cv::Vec3b* pOut;


        
        restart = false;
        // run a new algorithm or die
        while (!restart && !quit) {

            // start of frame
            auto start_frame = std::chrono::high_resolution_clock::now();
            
            bufStart = 0;

            // grab webcam frame

#ifdef USE_CAPTURE_THREAD
            {
                // avoid blocking capture 
                std::lock_guard<std::mutex> lock(mtx);
                if(!frame_queue.empty()){
                    frame = frame_queue.front();
                    frame_queue.pop();
                }else{
                    // no new frame, skip processing
                    //if(verbose) std::cout << "No new frame available" << std::endl;
                    continue;
                }
            }
#else           
            // this code blocks until a frame is captured
            cap >> frame; 
            if(frame.empty()){
                std::cerr<< "Capture failed" << std::endl;
                exit(0);
            }
#endif

        cv::Vec3b* pSrc = frame.ptr<cv::Vec3b>(0);
        cv::Vec3b* pBuf = lineBuffer.ptr<cv::Vec3b>(0);
        cv::Vec3b* pOut = renderBuf.ptr<cv::Vec3b>(0);

            auto core_start = std::chrono::high_resolution_clock::now();
            // try to optimize out .at<> calls
            if(  continuousBuffers ){

                const int cols = frame.cols;
                const int rows = frame.rows;
                const int framenum_plus1 = framenum + 1;
                const bool doNoise = (noise > 0.0 || auto_noise > 0.0);
                const int noiseThreshold = static_cast<int>(std::max(noise, auto_noise));

                // Row-major iteration for cache locality
                for (int r = 0; r < rows; ++r) {

                    const int* __restrict__ depthRow = pixDepth[r].data();
                    const int rowOffset = r * cols;

                    for (int c = 0; c < cols; ++c) {

                        const int z = depthRow[c];
                        const int pixelIndex = rowOffset + c;

                        pBuf[bufStart + (framenum % z)] = pSrc[pixelIndex];
                        pOut[pixelIndex] = pBuf[bufStart + (framenum_plus1 % z)];

                        bufStart += z;
                    }
                }

                // Separate noise pass (only when needed)
                if(doNoise){
                    const int totalPixels = rows * cols;
                    for (int i = 0; i < totalPixels; ++i) {
                        if((std::rand() % 100) < noiseThreshold){
                            const int q = std::rand() % 256;
                            pOut[i] = cv::Vec3b(q, q, q);
                        }
                    }
                }

            }else{
                // use buffer.at<cv::Vec3b>
    
                for(int c = 0; c < frame.cols; c++){
                
                    for(int r = 0; r < frame.rows; r++){

                        int z = pixDepth[r][c];

                        // stash this frame's pixel
                        lineBuffer.at<cv::Vec3b>(bufStart + (framenum % z), 0) = frame.at<cv::Vec3b>(r,c);

                        // pull older pixel
                        renderBuf.at<cv::Vec3b>(r,c) = lineBuffer.at<cv::Vec3b>(bufStart + ((framenum + 1)% z), 0);

                        bufStart += z;
                    }
                }
            }
            if(auto_noise > 0.0){
                auto_noise = auto_noise -1.0;
                if(auto_noise < 0.0) auto_noise = 0.0;
            }
            auto core_end = std::chrono::high_resolution_clock::now();

            auto start_render = std::chrono::high_resolution_clock::now();
            //     0: Flips the image vertically (around the x-axis).
            //     1: Flips the image horizontally (around the y-axis), creating a mirror effect.
            //    -1: Flips the image both horizontally and vertically (equivalent to a 180-degree rotation).

            // Upload to GPU and do post-processing there
            renderBuf.copyTo(renderGpu);

            if(mirrorHorizontal){
                cv::flip(renderGpu, renderGpu, 1);
            }
            if(mirrorVertical){
                cv::flip(renderGpu, renderGpu, 0);
            }

            cv::Size outputSize( dispWidth, dispHeight);
            cv::resize(renderGpu, finalGpu, outputSize);

            //TODO: apply filtering here? or before resizing?
            // int f = depthMax / 2;
            // f = (f % 2 == 0) ? f+1 : f;
            int f = 2 * (filterOption) + 1; // 0=>1, 1=>3, 2=>5, etc

            if(filterOption){
                cv::GaussianBlur(finalGpu, finalGpu, cv::Size(f, f), 0);
            }

            cv::imshow("slow glass", finalGpu);

            if(debugWindow){
                cv::imshow("debug", frame);
            }
            auto end_render = std::chrono::high_resolution_clock::now();

            framenum++;


            // Exit loop if 'q' is pressed
            int key = cv::waitKey(1); // losing at least a milllisecond :(

            switch(key){
                case 'Q':
                case 'q':
                case 27:
                    quit = true;
                    break;

                case 'r':
                    noise--;
                    if(noise < 0.0) noise = 0.0;
                    assertInRange(noise, 0.0, 100.0);
                break;
            
                case 'R':
                    noise++;
                    if(noise > 100.0) noise = 100.0;
                    assertInRange(noise, 0.0, 100.0);
                break;

                case 'N':
                    auto_noise = 100.0;
                break;

                case 'h':
                case 'H':
                    mirrorHorizontal = !mirrorHorizontal;
                    std::cout << "\nMirror horizontal " << (mirrorHorizontal ? "ON" : "OFF") << std::endl;
                    break;
                    
                case 'v':
                case 'V':
                    mirrorVertical = !mirrorVertical;
                    std::cout << "\nMirror vertical " << (mirrorVertical ? "ON" : "OFF") << std::endl;
                    break;   

                case 'a':
                case 'A':

                    // iterate through available algorithms
                    restart = true;
                    algorithm++;
                    if(algorithm > N_ALGORITHMS) algorithm = 1;
                    pEffOfEcksWye = algorithms[algorithm - 1];
                    if(verbose)std::cout << "\nAlgorithm " << algorithm << std::endl;

                    lineBuffer.release();
                    break;

                case '+':
                case '=':
                    param++;
                    if(param > depthMax) param = depthMax;

                    // iterate through available algorithms
                    restart = true;
                    
                    if(verbose) std::cout << "\nNew param: " << param << std::endl;

                    lineBuffer.release();
                    break;

                case '-':
                case '_':
                    param--;
                    if(param < 0) param = 0;

                    // iterate through available algorithms
                    restart = true;
                    
                    if(verbose) std::cout << "\nNew param: " << param << std::endl;

                    lineBuffer.release();
                    break;

                case 'd':
                case 'D':
                    depthMax -= 1;
                    if(depthMax < 1) depthMax = 1;
                    //std::cout << "Depth max decreased to " << depthMax << std::endl;
                    restart = true;
                    lineBuffer.release();
                    break;
                
                case 'U':
                case 'u':
                    depthMax += 1;
                    //std::cout << "Depth max increased to " << depthMax << std::endl;
                    restart = true;
                    lineBuffer.release();
                    break;

                case 'f':
                case 'F':
                    filterOption++;
                    if(filterOption > MAX_FILTERS) filterOption = 0;
                    std::cout << "\nFilter option set to " << filterOption << std::endl;
                    break;

                default:
                    //printf("Default key handler: keycode %d\n", key);
                    break; //really?

            }//end switch
                       
            once = true;

            if(timeout != 0.0){
                auto now =  std::chrono::steady_clock::now();
                std::chrono::duration<double> elapsed_seconds = now - beginTimeout;
                if(elapsed_seconds.count() >= timeout){
                    // time to change algo
                    restart = true;
                    algorithm++;
                    if(algorithm > N_ALGORITHMS) algorithm = 1;
                    pEffOfEcksWye = algorithms[algorithm - 1];
                    if(verbose) std::cout << "\nTimeout reached. Switching to algorithm " << algorithm << std::endl;
                    lineBuffer.release();
                }
            }

            // calculate FPS
            static double FPS_A[16] = {0.0};
            static double core_A[16] = {0.0};
            static double render_A[16] = {0.0};
            static int FPS_i = 0;

            if(verbose){
                auto end_frame = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> frame_duration = end_frame - start_frame;
                FPS_A[FPS_i++ % 16] = frame_duration.count();
                double FPS = 0.0;
                for(int i = 0; i < 16; i++) FPS += FPS_A[i];
                FPS /= 16.0;

                std::chrono::duration<double, std::milli> core_duration = core_end - core_start;
                core_A[FPS_i % 16] = core_duration.count();
                double CORE = 0.0;
                for(int i = 0; i < 16; i++) CORE += core_A[i];
                CORE /= 16.0;
                //std::cout << " \t\t\tcore: " << CORE << std::flush; ;

                std::chrono::duration<double, std::milli> render_duration = end_render - start_render;
                render_A[FPS_i % 16] = render_duration.count();
                double RENDER = 0.0;
                for(int i = 0; i < 16; i++) RENDER += render_A[i];
                RENDER /= 16.0;

                if(framenum < 16){
                    FPS = 1000;
                }

                 std::cout << "\r[" << framenum << "]\t\tFPS: "<< std::fixed << std::setprecision(0)  << 1000/FPS  
                                                 <<"  Loop: "<< FPS 
                                                 << " Core [" << std::fixed << std::setprecision(0) << CORE 
                                                 <<"] render["<< std::fixed << std::setprecision(0) << RENDER << "]"
                                                 << std::flush;
            }

            
        }// end while restart
    }// end while restart

#ifdef USE_CAPTURE_THREAD

    stop_flag.store(true);
    captureThread.join();

#endif

    return 0;
}

//////////////////////////////

int algo_cosX_x_cosY(int r, int c, int depth_param, double variation){

    return 1 + int(depthMax * cos(M_PI * c /camWidth - M_PI/2) * cos(M_PI * r/camHeight - M_PI/2)); 
}

int algo_Inv_cosX_x_cosY(int r, int c, int depth_param, double variation){

    return 1 + depthMax - int(depthMax * cos(M_PI * c /camWidth - M_PI/2) * cos(M_PI * r/camHeight - M_PI/2)); 
}

int algo_OG_param(int r, int c, int depth_param, double variation){
    if(depth_param < 1) depth_param = 1;
    if(depth_param > depthMax) depth_param = depthMax;
    return (r/depth_param)+1;
}

int algo_bottom_up(int r, int c, int depth_param, double variation){
    if(depth_param < 1) depth_param = 1;
    if(depth_param > depthMax) depth_param = depthMax;
    return ((camHeight - r)/depth_param)+1;
}

int algo_left_to_right(int r, int c, int depth_param, double variation){
    if(depth_param < 1) depth_param = 1;
    if(depth_param > depthMax) depth_param = depthMax;
    return ((c)/depth_param)+1;
}

int algo_right_to_left(int r, int c, int depth_param, double variation){
    if(depth_param < 1) depth_param = 1;
    if(depth_param > depthMax) depth_param = depthMax;
    return ((camWidth - c)/depth_param)+1;
}

int algo_cos_x(int r, int c, int depth_param, double variation){

    return 1 + int(depthMax * cos(M_PI * c /camWidth - M_PI/2)); 
}

int algo_cos_y(int r, int c, int depth_param, double variation){

    return 1 + int(depthMax * cos(M_PI * r/camHeight - M_PI/2)); 
}

int algo_inv_cos_x(int r, int c, int depth_param, double variation){

    return 1 + depthMax - int(depthMax * cos(M_PI * c /camWidth - M_PI/2)); 
}   
int algo_inv_cos_y(int r, int c, int depth_param, double variation){

    return 1 + depthMax - int(depthMax * cos(M_PI * r/camHeight - M_PI/2)); 
}

int algo_perlin(int r, int c, int depth_param, double variation){

    double scale = 0.009;
    int octaves = 1;
    static double persistence = 0.999;
    static siv::PerlinNoise::seed_type seed = std::rand() % 1000;;
    static siv::PerlinNoise perlin{ seed };

    // if( r == 0 && c == 0){
    //     std::srand(std::time(0)); // Seed the generator once
    //     seed = std::rand() % 1000;
    //     perlin;
    // }

    if(abs(persistence - variation) > 0.0001){
        // new algo
        persistence = variation;
        seed = std::rand() % 1000;
        perlin.reseed(seed);
        if(verbose) std::cout << "New perlin seed " << seed << ", Param: " << variation << std::endl;
    } 
    

    // static bool once = false;
    // if(!once){
    

    //     once = true;
    // }

    const double noise = perlin.normalizedOctave2D_01((r * scale), (c * scale), octaves, persistence);

    return  1 + depthMax * noise;      

}   
int algo_concentric(int r, int c, int depth_param, double variation){
    return algo_multi(r, c, depth_param, variation, 1);
}
int algo_sine(int r, int c, int depth_param, double variation){
    return algo_multi(r, c, depth_param, variation, 2);
}
int algo_sine2(int r, int c, int depth_param, double variation){
    return algo_multi(r, c, depth_param, variation, 3);
}
int algo_sine2_highfreq(int r, int c, int depth_param, double variation){
    return algo_multi(r, c, depth_param, variation, 4);
}
int algo_spiral(int r, int c, int depth_param, double variation){
    return algo_multi(r, c, depth_param, variation, 5);
}
int algo_multi(int r, int c, int depth_param, double variation, int mode){

    int mid = depthMax / 2;
    if(mid < 1) mid = 1;
    if(mid > depthMax) mid = depthMax;

    int cx = camWidth / 2;
    int cy = camHeight / 2;
    double rr =  1.0 * (r - cy) / camWidth;
    double cc =  1.0 * (c - cx) / camHeight;

    switch(mode){
        case 0:
            // random pattern
            return std::rand() % depthMax + 1;
            
            break;
        case 1:
            // concentric circles
            {

                double dist = sin(10 * sqrt((rr * rr) + (cc * cc))); // <-- orig

                return static_cast<int>(mid + dist * mid);//static_cast<uchar>(fmod(dist, 256));
            }
            break;
        case 2:
            // sine wave pattern
            {
                double val = mid + mid * sin(1 * M_PI * r / 32.0);
                if(val < 0) val = 0;
                if(val > 255) val = 255;
                return static_cast<int>(val);
            }
            break;
        case 3:
            // combined sine wave pattern
            {
                double val = mid + mid * sin(30 * M_PI * rr / 1.0) * cos(30 * M_PI * cc / 1.0);
                if(val < 0) val = 0;
                if(val > 255) val = 255;
                return static_cast<int>(val);
            }
            break;
            case 4:
            // combined sine wave pattern, higher frequency
            {
                double val = mid + mid * sin(4 * M_PI * rr / 2.0) * cos(4 * M_PI * cc / 2.0);   
                if(val < 0) val = 0;
                if(val > 255) val = 255;
                return static_cast<int>(val);
            }
            break;
        case 5:
        // spiral
            {
                int cx = camWidth / 2;
                int cy = camHeight / 2;
                double rr =  1.0 * (r - cy) / camWidth;
                double cc =  1.0 * (c - cx) / camHeight;
                double angle = atan2(rr, cc);
                double dist =  sin(220 * sqrt((rr * rr) + (cc * cc)) + 10 * angle); 
                return static_cast<int>(mid + dist * mid);

            }
            break;
        case 6:
        // fall thru
        default:
            // simple test pattern
            double xx = 1.0 * c / (camWidth / 8);
            double yy = 1.0 * r / (camHeight / 8);
            double n = sin(xx*yy);
            return static_cast<int>(255 *  n) % 256;
            break;
        }
 
}



//////////////////////////////

void keys(void){
    std::cout << "Interactive commands:" << std::endl;
    std::cout << "=====================" << std::endl;
    std::cout << "  'q' or 'Q' or ESC   Quit" << std::endl;
    std::cout << "  'a' or 'A'          Next algorithm" << std::endl;
    std::cout << "  '+' or '='          Increase algorithm parameter" << std::endl;
    std::cout << "  '-' or '_'          Decrease algorithm parameter" << std::endl;
    std::cout << "  'u' or 'U'          Increase maximum pixel history depth" << std::endl;
    std::cout << "  'd' or 'D'          Decrease maximum pixel history depth" << std::endl;
    std::cout << "  'h' or 'H'          Toggle mirror horizontal" << std::endl;
    std::cout << "  'v' or 'V'          Toggle mirror vertical" << std::endl;
    std::cout << "  'R'                 Increase noise injection" << std::endl;
    std::cout << "  'r'                 Decrease noise injection" << std::endl;
    std::cout << "  'N'                 Inject noise burst with automatic decay" << std::endl;
    std::cout << std::endl << std::endl;
}

void usage(void){
    std::cout << "Usage:" << progname << " [OPTIONS]" << std::endl;
    std::cout << std::endl;
    std::cout << "2D funhouse mirror effect with pixel history" << std::endl;
    std::cout << std::endl;
    std::cout << "Hit 'Q' to quit" << std::endl << std::endl;
    std::cout << "-v\tVerbose" << std::endl;
    std::cout << "-n\tName of input device. Defaults to /dev/video0" << std::endl;
    std::cout << "-g\tShow debug window (raw camera feed)" << std::endl;
    std::cout << "-d\tSet maximum pixel history depth [1-254]" << std::endl;
    std::cout << "-w\tSet camera width" << std::endl;
    std::cout << "-h\tSet camera height" << std::endl;
    std::cout << "-W\tSet output window width" << std::endl;
    std::cout << "-H\tSet output window height" << std::endl;
    std::cout << "-f\tSet filter to apply to output" << std::endl;
    std::cout << "-m\tMirror horizontally" << std::endl;
    std::cout << "-M\tMirror vertically" << std::endl;
    std::cout << "-t\tTimout in seconds. Interval to auto-increment algorithm" << std::endl;
    std::cout << "-a\tSelect algorithm 1 - "<< std::to_string(N_ALGORITHMS) << std::endl;
    std::cout << "-p\tOptional algorithm parameter - " << std::endl << std::endl;
    std::cout << "v4l2-ctl --list-formats-ext will show devices and capabilities" << std::endl;
    std::cout << std::endl;

    keys();
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
bool assertInRange(double n, double min, double max){

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

    while((opt = getopt(argc, argv, "a:gvd:w:W:h:H:f:p:n:t:mM")) != -1){
        switch(opt){

            case 'v':
            verbose = true;
            break;

            case 'n':
            video_device_name = optarg;
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
            assertInRange(depthMax, 1, 999);
              
            break;

            case 'h':
            camHeight = atoi(optarg);
            assertInRange(camHeight, 1, 8192);
            break;

            case 'H':
            dispHeight = atoi(optarg);
            assertInRange(dispHeight, 1, 8192);
            break;

            case 'w':
            camWidth = atoi(optarg);
            assertInRange(camWidth, 1, 8192);
            break;

            case 'W':
            dispWidth = atoi(optarg);
            assertInRange(dispWidth, 1, 8192);
            break;

            case 'f':
            filterOption = atoi(optarg);
            assertInRange(filterOption, 0, 999);
            break;
       
            case 't':
            timeout = atof(optarg);
            assertInRange(timeout, 0.0, FLT_MAX);
            break;

            case 'g':
            debugWindow = true;
            break;

            case 'm':
            mirrorHorizontal = true;
            break;  

            case 'M':
            mirrorVertical = true;
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

void mouse_callback(int event, int x, int y, int flags, void* userdata){
 
    if(event == cv::EVENT_LBUTTONDOWN){
        std::cout << "Mouse click at (" << x << ", " << y << ")" << std::endl;
    }
}

int enumerate_cameras(void){
    int dev_mask = 0;
    bool no_cameras = true;
    std::string dev_path = "/dev/";
    for (const auto & entry : std::filesystem::directory_iterator(dev_path)) {
        std::string filename = entry.path().filename().string();
        if (filename.find("video") == 0) {
            if(no_cameras){
                if(verbose)std::cout << "Available cameras:" << std::endl;
                no_cameras = false;
            }
            
            //dev_mask |= ( 1<< std::stoi(filename.back(), nullptr) );
            dev_mask |= ( 1 << '0' - filename.back() ); // feels icky
            if(verbose) std::cout  << entry.path() << std::endl;
        }

    }        
    if(no_cameras){
        if(verbose)std::cout << "No cameras found" << std::endl;   
    }
    return dev_mask;
}

