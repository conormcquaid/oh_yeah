
/*


   experiment with perlin noise, output to a thumbnail image

   command line parameters:
   param1: scale (e.g. 0.005)
   param2: octaves (e.g. 1)
   param3: persistence (e.g. 0.9999)

   e.g. ./pt 0.005 1 0.9999

   scale times width or height should be about 5 to 10 to get a good effect
   octaves 1 
   persistence  - near 1.0 for smooth, near 0.0 for rough
   ./pt 0.004 1 0.009

*/



#include <iostream>
#include "PerlinNoise.hpp"
#include <cstdlib> // For rand() and srand()
#include <ctime>   // For time()
#include <opencv2/opencv.hpp>
 
int main(int argc, char** argv) {
 
   std::srand(std::time(0)); // Seed the generator once
   const siv::PerlinNoise::seed_type seed = std::rand() % 100;
   const siv::PerlinNoise perlin{ seed };

   double scale = 0.009;
   int octaves = 1;
   double persistence = 0.9;
   int width = 1000;
   int height = 1000;

   cv::namedWindow("Thumbnail", cv::WINDOW_AUTOSIZE);
   cv::Mat thumbnail(height, width, CV_8UC1, cv::Scalar(0));

   if(argc > 2){
      scale = atof(argv[1]);
      octaves = atoi(argv[2]);
      persistence = atof(argv[3]);
   }

   double max = 0.0;
   double min = 1.0;

   for(int y = 0; y < width; ++y){

      for(int x = 0; x < height ; ++x){

         const double noise = perlin.normalizedOctave2D_01((y * scale ), (x * scale), octaves, persistence);

         thumbnail.at<uchar>(x,y) = 255 * noise;

         if(noise > max) max = noise;
         if(noise < min) min = noise;

         //std::cout << noise << "\t";

      }
      //std::cout << std::endl;
      



   }
   std::cout << std::endl << std::endl << "Max: " << max << " Min: " << min << std::endl << std::endl;


   cv::imshow("Thumbnail", thumbnail);
   while(true){
      int key = cv::waitKey(1);
      if (key == 'Q' || key == 'q' || key == 27) {
            break;
      }
   }
}
