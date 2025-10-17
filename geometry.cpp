
/*


   experiment with different geometric patterns

*/



#include <iostream>
#include <cstdlib> // For rand() and srand()
#include <ctime>   // For time()
#include <opencv2/opencv.hpp>
#include <math.h>


void mouse_callback(int event, int x, int y, int flags, void* userdata);


int main(int argc, char** argv) {
 
//   std::srand(std::time(0)); // Seed the generator once
//    const siv::PerlinNoise::seed_type seed = std::rand() % 100;
//    const siv::PerlinNoise perlin{ seed };

    double param1;
    int param2;
    double param3;
    int width = 256;
    int height = 256;

    int mode = 0;
    int iparam = 32;

    if(argc > 1){
       mode = atoi(argv[1]);
    }
    if(mode < 0 || mode > 6) mode = 0;

    if(argc >2){
        iparam = atoi(argv[2]); 
    }
    if(iparam < 1) iparam = 1;
    if(iparam > 255) iparam = 255;

    int front_buf;
    int back_buf;

    front_buf = 0;
    back_buf = 1 - front_buf;

    cv::namedWindow("Thumbnail", cv::WINDOW_AUTOSIZE);
    cv::Mat thumbnail(height, width, CV_8UC1, cv::Scalar(0));
    cv::Mat out(1280, 1280, CV_8UC1, cv::Scalar(0));

    cv::setMouseCallback("Thumbnail", mouse_callback, NULL);

    while(true){

        for(int r = 0; r < height; r++){
            for(int c = 0; c < width; c++){

                int cx = width / 2;
                int cy = height / 2;
                double rr =  1.0 * (r - cy) / width;
                double cc =  1.0 * (c - cx) / height;

                switch(mode){
                    case 0:
                        // simple test pattern
                        thumbnail.at<uchar>(r,c) = (r + c) % 256;
                        break;
                    case 1:
                        // concentric circles
                        {

                            double dist = sin(iparam * sqrt((rr * rr) + (cc * cc))); // <-- orig
                            //double dist = sqrt( ((r - cy)*2) * ((r - cy)*2) + ((c - cx)*2) * ((c - cx)*2) );
                            //double dist = sqrt( ((r - cy)) * ((r - cy))/2 + ((c - cx)) * ((c - cx))/2 );
                            thumbnail.at<uchar>(r,c) = static_cast<int>(65 + dist * 64);//static_cast<uchar>(fmod(dist, 256));
                            //std::cout << rr << " ";
                        }
                        break;
                    case 2:
                        // sine wave pattern
                        {
                            double val = 128 + 127 * sin((iparam / 10.0) * M_PI * r / 32);
                            if(val < 0) val = 0;
                            if(val > 255) val = 255;
                            thumbnail.at<uchar>(r,c) = (uchar)val;
                        }
                        break;
                    case 3:
                        // combined sine wave pattern
                        {
                            double val = 128 + 127 * sin((iparam / 10.0) * M_PI * rr / 1.0) * cos((iparam / 10.0) * M_PI * cc / 1.0);
                            if(val < 0) val = 0;
                            if(val > 255) val = 255;
                            thumbnail.at<uchar>(r,c) = (uchar)val;
                        }
                        break;
                        case 4:
                        // combined sine wave pattern, higher frequency
                        {
                            double val = 128 + 127 * sin((iparam / 1.0) * M_PI * rr / 4.0*1) * cos( (iparam / 1.0)* M_PI * cc / 4.0*1);   
                            if(val < 0) val = 0;
                            if(val > 255) val = 255;
                            thumbnail.at<uchar>(r,c) = (uchar)val;
                        }
                        break;
                    case 5:
                    // spiral
                        {
                            int cx = width / 2;
                            int cy = height / 2;
                            double rr =  0.5 * (r - cy) / width;
                            double cc =  0.5 * (c - cx) / height;
                            double angle = atan2(rr, cc);
                            double dist =  sin((22.0)* sqrt((rr * rr) + (cc * cc)) + 2.0 * angle); 
                            thumbnail.at<uchar>(r,c) = static_cast<int>(65 + dist * 64);

                            //std::cout << dist << " ";
                        }
                        break;
                    case 6:
                    // checkerboard
                    default:
                        // simple test pattern
                        double xx = 1.0 * c / (width / 8);
                        double yy = 1.0 * r / (height / 8);
                        double n = sin(xx*yy);
                        thumbnail.at<uchar>(r,c) = static_cast<int>(255 *  n) % 256;
                        break;




                }

                //double val = 128 + 127 * sin(2 * M_PI * r / 32.0) * cos(2 * M_PI * c / 32.0);









                // if(val < 0) val = 0;
                // if(val > 255) val = 255;
                // thumbnail.at<uchar>(r,c) = (uchar)val;
                // std::cout << val << " ";



                // simple test pattern
                //thumbnail.at<uchar>(r,c) = (r + c) % 256;
            }
        }




            //cv::imshow("Thumbnail", thumbnail);
            cv::resize(thumbnail, out, cv::Size(out.rows, out.cols));
            cv::imshow("Thumbnail", out);

            // block waiting for keypress
            int key = cv::waitKey(0);
            if(key == 'p'){
                iparam--;
                if(iparam < 1) iparam = 1;
                
            }
            if(key == 'P'){
                iparam++;
                if(iparam > 255) iparam = 255;
            }
            if (key == 'Q' || key == 'q' || key == 27) {
                exit(0);
            }
            std::cout << "Parameter : " << iparam << std::endl;
        }

    
    return 0;

}

void mouse_callback(int event, int x, int y, int flags, void* userdata){
 
    if(event == cv::EVENT_LBUTTONDOWN){
        std::cout << "Mouse click at (" << x << ", " << y << ")" << std::endl;
    }
}