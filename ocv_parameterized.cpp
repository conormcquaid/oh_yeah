#include <opencv2/opencv.hpp>
#include <cmath>


const int camWidth = 640;
const int camHeight = 480;
const int dispWidth = 1080;
const int dispHeight = 720;

int framenum;

// sum of 1st n integers
int sumn(int n){

   if(n == 0){
      return 0;
   }else{
      return (n-1) * n / 2;
   }
}

// depth is the size of the history per row
int Depth(int n){

    return floor(n/2);
}
// line start is the first row in linebuffer corresponding to source row n
int LineStart(int n){

    if(n==0){
        return 0;
    }else{
        return floor(n*n/4);
    }
}

int repSum(int line, int rep){
    int sum = 0;
    for(int i =0; i < line; i++){
        sum += floor(i/rep) + 1;
    }
    return sum;
}

int main(int argc, char* argv[]) {

    int nReps;

    if(argc == 2){
        nReps = atoi(argv[1]);
    }
    if(nReps == 0) nReps = 1;
    
    // 1. Open the default camera (index 0)
    cv::VideoCapture cap(0); 

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open webcam." << std::endl;
        return -1;
    }
    
    //cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    //cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  camWidth);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, camHeight);


    cv::Mat frame;
    cv::namedWindow("CORRIDOR", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("debug",    cv::WINDOW_AUTOSIZE);

    // Read a new frame from the camera
     cap >> frame;
     if (frame.empty()) {
         std::cerr << "Error: Frame is empty." << std::endl;
         exit(0);
     }    
     
       
    int mtype = frame.type();        
    
    cv::Mat lineBuffer(sumn(camHeight+1), camWidth, mtype);
     
    cv::Mat buf2(camHeight, camWidth, mtype);
    
    cv::Mat finalFrame;
    
    int once = 0;

    framenum = 0;

    std::cout << "FIXME: make efficient linebuffer allocation" << std::endl;

    for(int i = 0; i < camHeight; i++){

        std::cout << i << "," << repSum(i, nReps) << std::endl;
    }

   

    while (true) {
        // Read a new frame from the camera
        cap >> frame; 
        
        if(!once){         
            std::cout << "camera props. rows: " << frame.rows << ", cols: " << frame.cols << std::endl;
        }
  

        if (frame.empty()) {
            std::cerr << "Error: Frame is empty." << std::endl;
            break;
        }

        // --- 2. Transformation ---
        // Example transformation: Convert to grayscale and blur
  //      cv::Mat processed_frame;
  //      cv::cvtColor(frame, processed_frame, cv::COLOR_BGR2GRAY);
  //      cv::GaussianBlur(processed_frame, processed_frame, cv::Size(5, 5), 0);

    //      arrayCopy(cam.pixels,  myline*width, lineBuffer, (s + framenum%(i+1))*width, width);

       
    //    arrayCopy(lineBuffer,  (s + (framenum+1)%(myline+1))*width, pixels, myline * width, width);
       
       
        for(int i = 0; i < camHeight; i++){  // for each row

            int depth = floor(i / nReps) + 1;
            int lineStart = repSum(i, nReps);

            // push into history
            frame.row(i).copyTo(lineBuffer.row((lineStart + framenum%(depth))));
            
            //suck history into display
            lineBuffer.row((lineStart + (framenum+1)%(depth))).copyTo(buf2.row(i));
        }
        

        // invert frame
        // for(int i = 0; i < camHeight; i++){
        //    buf2.row(i) = frame.row(i);
        //    frame.row(i).copyTo(buf2.row(camHeight - i -1));
        // }
        
        cv::Size outputSize(dispWidth, dispHeight);
        //cv::Size outputSize(camWidth, camHeight);
        //cv::resize(processed_frame, finalFrame, outputSize);

        int f = nReps / 2;
        f = (f % 2 == 0) ? f+1 : f;

       cv::GaussianBlur(buf2, buf2, cv::Size(f, f), 0);

     //  cv::Mat tmp;
     //  cv::bilateralFilter(buf2, tmp, 5, 150, 150, cv::BORDER_DEFAULT);
      // buf2 = tmp;


        cv::resize(buf2, finalFrame, outputSize);
        
        if(!once){
         
         std::cout << "Final frame props. rows: " << frame.rows << ", cols: " << frame.cols  << std::endl;
         std::cout << "buf2  frame props. rows: " << buf2.rows  << ", cols: "  << buf2.cols  << std::endl;
         
         }
        
        cv::imshow("CORRIDOR", finalFrame);

        cv::imshow("debug", frame);

        // Exit loop if 'q' is pressed
        if (cv::waitKey(1) == 'q') {
            break;
        }
    
    
       framenum++;
       once = 1;
       
    }// true

    return 0;
}

