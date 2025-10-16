
/*


   experiment with perlin WORMs - essentailly, perlin noise with a bandpass
   output to a thumbnail image

   command line parameters:
   param1: min threshold
   param2: max threshold

   keys:
   Q - quit
   n/N - decrease/increase min threshold
   x/X - decrease/increase max threshold
   s/S - decrease/increase scale
   p/P - decrease/increase persistence
   r/R - new random seed

*/



#include <iostream>
#include "PerlinNoise.hpp"
#include <cstdlib> // For rand() and srand()
#include <ctime>   // For time()
#include <opencv2/opencv.hpp>

double thresh_min = 0.25;
double thresh_max = 0.75;

 
int main(int argc, char** argv) {
 
   std::srand(std::time(0)); // Seed the generator once
   siv::PerlinNoise::seed_type seed = std::rand() % 100;
   siv::PerlinNoise perlin{ seed };

   double scale = 0.009;
   int octaves = 1;
   double persistence = 0.9;
   int width = 512;
   int height = 512;

   cv::namedWindow("Thumbnail", cv::WINDOW_AUTOSIZE);
   cv::Mat thumbnail(height, width, CV_8UC1, cv::Scalar(0));

//    if(argc > 2){
//       param1 = atof(argv[1]);
//       param2 = atoi(argv[2]);
//       param3 = atof(argv[3]);
//    }
    if(argc > 2){
      thresh_min = atof(argv[1]);
      thresh_max = atof(argv[2]);
    }

   double max = 0.0;
   double min = 1.0;

   double* pNoise = new double[width * height];

    while(true){

        std::cout << "Perlin( " << scale << ", " << octaves << ", " << persistence << " )" << std::endl;
        std::cout << "Thresh( " << thresh_min << ", " << thresh_max << " )" << std::endl;

        // first pass : calculate perlin noise, establish min, max
    for(int y = 0; y < width; ++y){

        for(int x = 0; x < height ; ++x){

            const double noise = perlin.normalizedOctave2D_01((y * scale ), (x * scale), octaves, persistence);

            pNoise[x + width * y] = noise;

            if(noise > max) max = noise;
            if(noise < min) min = noise;
        }
    }
    // now, given min, max, render worms
    for(int y = 0; y < width; ++y){

        for(int x = 0; x < height ; ++x){

            if(pNoise[x + width * y] > (thresh_min * min)  && pNoise[x + width * y] < (thresh_max * max )){
            //if(pNoise[x + width * y] > (thresh_min * min)  && pNoise[x + width * y] < (thresh_max * max)){
                thumbnail.at<uchar>(x,y) = 255 * pNoise[x + width * y];
            }else{
                thumbnail.at<uchar>(x,y) = 255 - 255 * pNoise[x + width * y];
            }
        }
    }

    std::cout << "Max: " << max << " Min: " << min << std::endl<<"================================" << std::endl;


    cv::imshow("Thumbnail", thumbnail);

    int key = cv::waitKey(0);

      if (key == 'Q' || key == 'q' || key == 27) {
            break;
      }
        if(key == 'N'){
            thresh_min += 0.05;
            if(thresh_min > 1.0) thresh_min = 1.0;
            //std::cout << "thresh_min: " << thresh_min << std::endl;
        }
        if(key == 'n'){
            thresh_min -= 0.05;
            if(thresh_min < 0.0) thresh_min = 0.0;
            //std::cout << "thresh_min: " << thresh_min << std::endl;
        }
        if(key == 'x'){
            thresh_max += 0.05;
            if(thresh_max > 1.0) thresh_max = 1.0;
            //std::cout << "thresh_max: " << thresh_max << std::endl; 
        }
        if(key == 'X'){
            thresh_max += 0.05;
            if(thresh_max > 1.0) thresh_max = 1.0;
            //std::cout << "thresh_max: " << thresh_max << std::endl; 
        }
        if(key == 'p'){
            persistence -= 0.005;
            if(persistence < 0.0) persistence = 0.0;
            //std::cout << "persistence: " << persistence << std::endl; 
        }
            if(key == 'P'){
            persistence += 0.05;
            if(persistence > 1.0) persistence = 1.0;
            //std::cout << "persistence: " << persistence << std::endl; 
        }

        if(key == 's'){
            scale -= 0.005;
            if(scale < 0.0  ) scale = 0.0;
            //std::cout << "scale: " << scale << std::endl; 
        }
        if(key == 'S'){
            scale += 0.0005;
            if(scale > 1.0) scale = 1.0;
            //std::cout << "scale: " << scale << std::endl; 
        }
        if(key == 'r' || key == 'R'){
            seed = std::rand() % 100;
            perlin.reseed(seed);
            std::cout << "New seed: " << seed << std::endl; 
        }
   }




   delete(pNoise);
}
