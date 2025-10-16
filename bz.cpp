
/*


   experiment Byeltzev-Zhabotinsky reaction diffusion pattern generator

   intending to generate worm-like patterns. Result, meh.

   Keypress required to create next generation

   ~50 generations before anything happens. Some grayscale at this point.
   Shortly thereafter, output is usual BZ and fully saturated black/white.

*/



#include <iostream>
#include <cstdlib> // For rand() and srand()
#include <ctime>   // For time()
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <mutex>

class DoubleBuffered2D {
public:
    // Constructor to initialize two buffers with a specific size
    DoubleBuffered2D(size_t width, size_t height)
        : m_width(width), m_height(height) {
        m_buffer1 = std::make_unique<std::vector<float>>(width * height);
        m_buffer2 = std::make_unique<std::vector<float>>(width * height);
        m_front_buffer = m_buffer1.get();
        m_back_buffer = m_buffer2.get();
    }

    // Safely write a value to the back buffer
    void writeBack(size_t x, size_t y, float value) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (x < m_width && y < m_height) {
            (*m_back_buffer)[y * m_width + x] = value;
        }
    }

    void writeFront(size_t x, size_t y, float value) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (x < m_width && y < m_height) {
            (*m_front_buffer)[y * m_width + x] = value;
        }
    }
    // Safely read a value from the front buffer
    float readFront(size_t x, size_t y) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (x < m_width && y < m_height) {
            return (*m_front_buffer)[y * m_width + x];
        }
        return 0.0f;
    }

    float readBack(size_t x, size_t y) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (x < m_width && y < m_height) {
            return (*m_back_buffer)[y * m_width + x];
        }
        return 0.0f;
    }
    // Swap the front and back buffers
    void swap() {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::swap(m_front_buffer, m_back_buffer);
    }

private:
    size_t m_width, m_height;
    std::unique_ptr<std::vector<float>> m_buffer1, m_buffer2;
    std::vector<float>* m_front_buffer;
    std::vector<float>* m_back_buffer;
    mutable std::mutex m_mutex; // Protects access during read/write/swap
};


void mouse_callback(int event, int x, int y, int flags, void* userdata);



int main(int argc, char** argv) {
 
   std::srand(std::time(0)); // Seed the generator once
//    const siv::PerlinNoise::seed_type seed = std::rand() % 100;
//    const siv::PerlinNoise perlin{ seed };

    double param1;
    int param2;
    double param3;
    int width = 128;
    int height = 128;

    DoubleBuffered2D A_grid(width, height);
    DoubleBuffered2D B_grid(width, height);
    DoubleBuffered2D C_grid(width, height);

    // float a[width][height][2];
    // float b[width][height][2];
    // float c[width][height][2]; 

    int front_buf;
    int back_buf;

    front_buf = 0;
    back_buf = 1 - front_buf;

   cv::namedWindow("Thumbnail", cv::WINDOW_AUTOSIZE);
   cv::Mat thumbnail(height, width, CV_8UC1, cv::Scalar(0));
   cv::Mat out(1024, 1024, CV_8UC1, cv::Scalar(0));

   cv::setMouseCallback("Thumbnail", mouse_callback, NULL);

//    if(argc > 2){
//       param1 = atof(argv[1]);
//       param2 = atoi(argv[2]);
//       param3 = atof(argv[3]);
//    }

    // seed buffer 0
    for(int i = 0; i < width; ++i){
      for(int j = 0; j < height; ++j){
         A_grid.writeBack(i, j, std::rand() % 256 / 256.0f);
         B_grid.writeBack(i, j, std::rand() % 256 / 256.0f);
         C_grid.writeBack(i, j, std::rand() % 256 / 256.0f);
      }
    }
    // A_grid.swap();
    // B_grid.swap();
    // C_grid.swap();
    int generation = 0;

    float A,B,C;
    while(true){

        for(int y = 0; y < width ; ++y){
            for(int x = 0; x < height  ; ++x){

                A = B = C = 0.0;

                for(int dx = -1; dx <= 1; ++dx){
                    for(int dy = -1; dy <= 1; ++dy){

                        int xx = (x + dx + width) % width;
                        int yy = (y + dy + height) % height;

                        A += A_grid.readBack(xx, yy);
                        B += B_grid.readBack(xx, yy);
                        C += C_grid.readBack(xx, yy);

                    }
                }
                A /= 9.0;
                B /= 9.0;
                C /= 9.0;

                A_grid.writeFront(x, y, std::min(255.0f, std::max(0.0f, A + (A*(B - C)) )));//* 0.01f)));
                B_grid.writeFront(x, y, std::min(255.0f, std::max(0.0f, B + (B*(C - A)) )));//* 0.01f)));
                C_grid.writeFront(x, y, std::min(255.0f, std::max(0.0f, C + (C*(A - B)) )));//* 0.01f)));

                thumbnail.at<uchar>(y,x) = (uchar)A_grid.readFront(x, y);

            }
            
        }
        A_grid.swap();
        B_grid.swap();
        C_grid.swap();  

        //cv::waitKey(0);



        //cv::imshow("Thumbnail", thumbnail);
        cv::resize(thumbnail, out, cv::Size(out.rows, out.cols));
        cv::imshow("Thumbnail", out);

        int key = cv::waitKey(0);
        if (key == 'Q' || key == 'q' || key == 27) {
            break;
        }

        generation++;
        //if(generation%10 == 0) 
        std::cout << "Generation " << generation << "\r" << std::flush;
    }
    return 0;

}

void mouse_callback(int event, int x, int y, int flags, void* userdata){
 
    if(event == cv::EVENT_LBUTTONDOWN){
        std::cout << "Mouse click at (" << x << ", " << y << ")" << std::endl;
    }
}