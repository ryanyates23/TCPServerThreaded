#define __CL_ENABLE_EXCEPTIONS

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include "cl.hpp"
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

#define WIDTH 1280
#define HEIGHT 720

#define NUM_BUFFERS 3

//#define WIDTH 1024
//#define HEIGHT 576

//using namespace std;

//global variables between threads

unsigned char rBuf1[WIDTH*HEIGHT];
unsigned char rBuf2[WIDTH*HEIGHT];
unsigned char rBuf3[WIDTH*HEIGHT];
int rHead, rTail, rCount;

unsigned char sBuf1[WIDTH*HEIGHT];
unsigned char sBuf2[WIDTH*HEIGHT];
unsigned char sBuf3[WIDTH*HEIGHT];
int sHead, sTail, sCount;

std::mutex rMut1, rMut2, rMut3;
std::mutex sMut1, sMut2, sMut3;



int sockfr, sockfs, receiveSocket, sendSocket, portnoR, portnoS;


void error(const char *msg)
{
    perror(msg);
    exit(1);
}

void TCPSend(void)
{
    int totalBytesWritten = 0;
    int vecSize = WIDTH*HEIGHT;
    int n;
    
    for(;;)
    {
        //if(sCount > 0)
        {
            totalBytesWritten = 0;
            //if(sTail == sHead) sTail = (++sTail) % NUM_BUFFERS;
            if(sTail == 0)
            {
                sMut1.lock();
                while(totalBytesWritten != vecSize)
                {
                    n = (int)write(sendSocket,&sBuf1[totalBytesWritten],vecSize);
                    //cout << "Bytes Written: " << n << endl;
                    if (n < 0)
                        error("ERROR writing to socket");
                    totalBytesWritten += n;
                }
                sMut1.unlock();
            }
            else if(sTail == 1)
            {
                sMut2.lock();
                while(totalBytesWritten != vecSize)
                {
                    n = (int)write(sendSocket,&sBuf2[totalBytesWritten],vecSize);
                    //cout << "Bytes Written: " << n << endl;
                    if (n < 0)
                        error("ERROR writing to socket");
                    totalBytesWritten += n;
                }
                sMut2.unlock();
            }
            else if(sTail == 2)
            {
                sMut3.lock();
                while(totalBytesWritten != vecSize)
                {
                    n = (int)write(sendSocket,&sBuf3[totalBytesWritten],vecSize);
                    //cout << "Bytes Written: " << n << endl;
                    if (n < 0)
                        error("ERROR writing to socket");
                    totalBytesWritten += n;
                }
                sMut3.unlock();
            }
            sTail = (++sTail) % NUM_BUFFERS;
            sCount--;
        }
    }
}

void TCPReceive(void)
{
    int totalBytesRead = 0;
    int vecSize = WIDTH*HEIGHT;
    int n;
    
    
    for(;;)
    {
        totalBytesRead = 0;
        //if(rHead == rTail) rHead = (++rHead) % NUM_BUFFERS;
        if(rHead == 0 )
        {
            rMut1.lock();
            while(totalBytesRead != vecSize)
            {
                n = (int)read(receiveSocket,&rBuf1[totalBytesRead],vecSize - totalBytesRead);
                if (n < 0) error("ERROR reading from socket");
                totalBytesRead += n;
            }
            rMut1.unlock();
        }
        else if(rHead == 1)
        {
            rMut2.lock();
            while(totalBytesRead != vecSize)
            {
                n = (int)read(receiveSocket,&rBuf2[totalBytesRead],vecSize - totalBytesRead);
                if (n < 0) error("ERROR reading from socket");
                totalBytesRead += n;
            }
            rMut2.unlock();
        }
        else if(rHead == 2)
        {
            rMut3.lock();
            while(totalBytesRead != vecSize)
            {
                n = (int)read(receiveSocket,&rBuf3[totalBytesRead],vecSize - totalBytesRead);
                if (n < 0) error("ERROR reading from socket");
                totalBytesRead += n;
            }
            rMut3.unlock();
        }
        rHead = (++rHead) % NUM_BUFFERS;
    }
    
    
}


int main(int argc, char *argv[])
{
    //-------------BASIC INIT--------------------//
    //    int argc = 51717;
    //    char *argv = "localhost";
    int x, y;
    int frame = 0;
    unsigned char configByte = 0;
    //int displayFrame = 0;
    int filterType = 0; //0 = none, 1 = 3x3mean, 2 = 3x3 median, 3 = 5x5 median, 4 = 3x3median 2 pass
    int enSobelEdge = 0;
    int enBinarise = 0;
    int enCombine = 0;
    int enCombineFiltering = 0;
    int abort = 0;
    int width = WIDTH;
    int height = HEIGHT;
    
    int sum, max;
    float mean, thresh;
    float threshFactor = 0.70;
    float fps;
    rHead = 0;
    rTail = 0;
    
    
    
    
    //--------------OPENCV INIT---------------//
    int vecSize = WIDTH*HEIGHT;
    std::vector<unsigned char> h_frame_in(vecSize);
    std::vector<unsigned char> h_frame_out(vecSize);
    
    //--------------OPENCL INIT--------------------//
    cl::Program program;
    cl::Context context;
    
    //This section sets up a variable to accept platform info, then gets it
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    
    //This selects the default platform
    cl::Platform platform = platforms.front();
    printf("Generated Platform\n");
    
    //This sets up a variable for the devices and then gets them
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    
    //This selects the default device
    cl::Device device = devices.front();
    printf("Established Device\n");
    
    std::cout << device.getInfo<CL_DEVICE_TYPE>();
    std::cout << CL_DEVICE_TYPE_GPU << std::endl;
    std::cout << "CL_Image Support: " << device.getInfo<CL_DEVICE_IMAGE_SUPPORT>() << std::endl;
    std::cout << "Number of Compute Units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
    std::cout << "Global Memory Size (MB): " <<device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()/1000000 << std::endl;
    std::cout << "Local Memory Size (KB): " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()/1024 << std::endl;
    std::cout << "Max Clock Frequency (MHz): " <<device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << std::endl;
    
    //This reads in the kernel file
    std::ifstream kernelFile("kernels.cl");
    std::string src(std::istreambuf_iterator<char>(kernelFile), (std::istreambuf_iterator<char>()));
    printf("Read Kernel File\n");
    
    //This turns the string input into a source for kernels
    cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));
    
    //This creates an OpenCL context and a program from the sources
    context = cl::Context(device);
    printf("Created Context\n");
    program = cl::Program(context, sources);
    
    //This tries to build the kernels and outputs a build log if it fails miserably
    try {
        cl::Error err = program.build();
        printf("Built Program\n");
    }
    catch (cl::Error& e)
    {
        if (e.err() == CL_BUILD_PROGRAM_FAILURE)
        {
            for (cl::Device dev : devices)
            {
                // Check the build status
                cl_build_status status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
                if (status != CL_BUILD_ERROR)
                    continue;
                
                // Get the build log
                std::string name = dev.getInfo<CL_DEVICE_NAME>();
                std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
                std::cerr << "Build log for " << name << ":" << std::endl
                << buildlog << std::endl;
            }
        }
        else
        {
            throw e;
        }
        
    }
    
    cl::CommandQueue queue(context, device);
    
    cl::Buffer d_frame_in(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * h_frame_in.size(), h_frame_in.data());
    cl::Buffer d_original_frame(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char)*h_frame_in.size(), h_frame_in.data());
    cl::Buffer d_frame_out(context, CL_MEM_READ_WRITE, sizeof(unsigned char)*h_frame_out.size());
    
    cl::Kernel meanKernel(program, "MeanFilter");
    cl::Kernel medianKernel3(program, "MedianFilter3");
    cl::Kernel medianKernel5(program, "MedianFilter5");
    cl::Kernel sobelKernel(program, "SobelEdge");
    cl::Kernel binKernel(program, "Binarise");
    cl::Kernel combineKernel(program, "CombineImages");
    
    
    //------------TCP SERVER INIT--------------//
    
    //    int sockfr, sockfs, receiveSocket, sendSocket, portnoR, portnoS;
    socklen_t clilen;
    //char buffer[256];
    
    //set up socket to recieve frames on
    struct sockaddr_in serv_addr, cli_addr;
    if (argc < 2) {
        fprintf(stderr,"ERROR, no port provided\n");
        exit(1);
    }
    sockfr = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfr < 0)
        error("ERROR opening socket");
    bzero((char *) &serv_addr, sizeof(serv_addr));
    //portnoR = atoi(argv[1]);
    portnoR = 12345;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(portnoR);
    if (bind(sockfr, (struct sockaddr *) &serv_addr,
             sizeof(serv_addr)) < 0)
        error("ERROR on binding");
    listen(sockfr,5);
    clilen = sizeof(cli_addr);
    receiveSocket = accept(sockfr,
                           (struct sockaddr *) &cli_addr,
                           &clilen);
    if (receiveSocket < 0)
        error("ERROR on accept");
    
    //set up socket to send frames on
    sockfs = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfs < 0)
        error("ERROR Opening Socket");
    bzero((char* ) &serv_addr, sizeof(serv_addr));
    portnoS = 54321;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(portnoS);
    if (bind(sockfs, (struct sockaddr *) &serv_addr,
             sizeof(serv_addr)) < 0)
        error("ERROR on binding");
    listen(sockfs, 5);
    clilen = sizeof(cli_addr);
    sendSocket = accept(sockfs,
                        (struct sockaddr *) &cli_addr,
                        &clilen);
    if (receiveSocket < 0)
        error("ERROR on accept");
    
    //Start threads
    std::thread ReceiveThread(TCPReceive);
    std::thread SendThread(TCPSend);
    
    
    while(1)
    {
        
        //----------------LOAD IMAGE FROM RECEIVE BUFFER----------------------//
        
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        //if(rTail == rHead) rTail = (++rTail) % NUM_BUFFERS;
        if(rTail == 0)
        {
            rMut1.lock();
            for (y = 0; y < HEIGHT; y++)
            {
                for (x = 0; x < WIDTH-1; x++)
                {
                    h_frame_in[y*WIDTH + x] = rBuf1[y*WIDTH + x];
                }
            }
            rMut1.unlock();
        }
        else if(rTail == 1)
        {
            rMut2.lock();
            for (y = 0; y < HEIGHT; y++)
            {
                for (x = 0; x < WIDTH-1; x++)
                {
                    h_frame_in[y*WIDTH + x] = rBuf2[y*WIDTH + x];
                }
            }
            rMut2.unlock();
        }
        else if(rTail == 2)
        {
            rMut3.lock();
            for (y = 0; y < HEIGHT; y++)
            {
                for (x = 0; x < WIDTH-1; x++)
                {
                    h_frame_in[y*WIDTH + x] = rBuf3[y*WIDTH + x];
                }
            }
            rMut3.unlock();
        }
        rTail = (++rTail) % 3;
        
        
        //----------Get and decode decode byte-------------//
        configByte = h_frame_in[0];
        filterType = (0x03) & configByte;
        enSobelEdge = ((0x01 << 2) & configByte) >> 2;
        enBinarise = ((0x01 << 3) & configByte) >> 3;
        enCombine = ((0x01 << 4) & configByte) >> 4;
        enCombineFiltering = ((0x01 << 5) & configByte) >> 5;
        abort = ((0x01 << 6) & configByte) >> 6;
        
        if(abort) break;
        
        //----------------PROCESSING----------------------//
        
        queue.enqueueWriteBuffer(d_frame_in, CL_TRUE, 0, sizeof(unsigned char)*h_frame_in.size(), h_frame_in.data());
        queue.enqueueWriteBuffer(d_original_frame, CL_FALSE, 0, sizeof(unsigned char)*h_frame_in.size(), h_frame_in.data());
        
        if (filterType == 0)
        {
            queue.enqueueWriteBuffer(d_frame_out, CL_TRUE, 0, sizeof(unsigned char)*h_frame_in.size(), h_frame_in.data());
        }
        
        if (filterType == 1)
        {
            
            meanKernel.setArg(0, d_frame_in);
            meanKernel.setArg(1, d_frame_out);
            meanKernel.setArg(2, width);
            meanKernel.setArg(3, height);
            
            queue.enqueueNDRangeKernel(meanKernel, cl::NullRange, cl::NDRange(width, height));
            
            //queue.enqueueReadBuffer(d_frame_out, CL_TRUE, 0, sizeof(unsigned char)*h_frame_out.size(), h_frame_out.data());
            //queue.finish();
            //h_frame_in = h_frame_out;
        }
        else
        {
            //h_frame_out = h_frame_in;
        }
        if (filterType == 2)
        {
            
            medianKernel3.setArg(0, d_frame_in);
            medianKernel3.setArg(1, d_frame_out);
            medianKernel3.setArg(2, width);
            medianKernel3.setArg(3, height);
            
            queue.enqueueNDRangeKernel(medianKernel3, cl::NullRange, cl::NDRange(width, height),cl::NullRange, NULL, NULL);
            
            //queue.enqueueReadBuffer(d_frame_out, CL_TRUE, 0, sizeof(unsigned char)*h_frame_out.size(), h_frame_out.data());
            queue.finish();
            //h_frame_in = h_frame_out;
        }
        else
        {
            //h_frame_out = h_frame_in;
        }
        if(filterType == 3)
        {
            
            medianKernel5.setArg(0, d_frame_in);
            medianKernel5.setArg(1, d_frame_out);
            medianKernel5.setArg(2, width);
            medianKernel5.setArg(3, height);
            
            queue.enqueueNDRangeKernel(medianKernel5, cl::NullRange, cl::NDRange(width, height),cl::NullRange, NULL, NULL);
            queue.finish();
            if(enCombine == 1 && enCombineFiltering == 1)
            {
                queue.enqueueCopyBuffer(d_frame_out, d_original_frame, 0, 0, vecSize);
            }
        }
        else
        {
            //h_frame_out = h_frame_in;
        }
        if (enSobelEdge == 1)
        {
            
            sobelKernel.setArg(0, d_frame_out);
            sobelKernel.setArg(1, d_frame_in);
            sobelKernel.setArg(2, width);
            sobelKernel.setArg(3, height);
            
            queue.enqueueNDRangeKernel(sobelKernel, cl::NullRange, cl::NDRange(width, height),cl::NullRange, NULL, NULL);
            queue.finish();
        }
        else
        {
            h_frame_out = h_frame_in;
        }
        if(enBinarise == 1)
        {
            queue.enqueueReadBuffer(d_frame_in, CL_TRUE, 0, sizeof(unsigned char)*h_frame_in.size(), h_frame_in.data());
            
            sum = 0;
            max = 0;
            for (y = 0; y < height; y++)
            {
                for (x = 0; x < width; x++)
                {
                    sum += h_frame_in[y*width + x];
                    if(h_frame_in[y*width + x] > max) max = h_frame_in[y*width + x];
                }
            }
            
            mean = sum/(width*height);
            thresh = mean+((max-mean)*threshFactor);
            
            
            binKernel.setArg(0, d_frame_in);
            binKernel.setArg(1, d_frame_out);
            binKernel.setArg(2, width);
            binKernel.setArg(3, height);
            binKernel.setArg(4, (unsigned char)thresh);
            
            queue.enqueueNDRangeKernel(binKernel, cl::NullRange, cl::NDRange(width, height),cl::NullRange, NULL, NULL);
            queue.finish();
        }
        else
        {
            //h_frame_out = h_frame_in;
        }
        
        if (enCombine == 1 && enBinarise == 1)
        {
            
            combineKernel.setArg(0, d_frame_out);
            combineKernel.setArg(1, d_original_frame);
            combineKernel.setArg(2, d_frame_in);
            combineKernel.setArg(3, width);
            combineKernel.setArg(4, height);
            
            queue.enqueueNDRangeKernel(combineKernel, cl::NullRange, cl::NDRange(width, height),cl::NullRange, NULL, NULL);
            queue.finish();
        }
        else if (enCombine == 1 && enBinarise == 0)
        {
            
            combineKernel.setArg(0, d_frame_in);
            combineKernel.setArg(1, d_original_frame);
            combineKernel.setArg(2, d_frame_out);
            combineKernel.setArg(3, width);
            combineKernel.setArg(4, height);
            
            queue.enqueueNDRangeKernel(combineKernel, cl::NullRange, cl::NDRange(width, height),cl::NullRange, NULL, NULL);
            queue.finish();
        }
        
        if ((enBinarise == 0 && enCombine == 0) || (enBinarise == 1 && enCombine == 1))
        {
            queue.enqueueReadBuffer(d_frame_in, CL_TRUE, 0, sizeof(unsigned char)*h_frame_out.size(), h_frame_out.data());
        }
        else
        {
            queue.enqueueReadBuffer(d_frame_out, CL_TRUE, 0, sizeof(unsigned char)*h_frame_out.size(), h_frame_out.data());
        }
        
        //----------------SAVE IMAGE TO SEND BUFFER----------------------//
        
        //if(sHead == sTail) sHead = (++sHead) % NUM_BUFFERS;
        if(sHead == 0)
        {
            sMut1.lock();
            for (y = 0; y < HEIGHT; y++)
            {
                for (x = 0; x < WIDTH-1; x++)
                {
                    sBuf1[y*WIDTH + x] = h_frame_out[y*WIDTH + x];
                }
            }
            sMut1.unlock();
        }
        else if(sHead == 1)
        {
            sMut2.lock();
            for (y = 0; y < HEIGHT; y++)
            {
                for (x = 0; x < WIDTH-1; x++)
                {
                    sBuf2[y*WIDTH + x] = h_frame_out[y*WIDTH + x];
                }
            }
            sMut2.unlock();
        }
        else if(sHead == 2)
        {
            sMut3.lock();
            for (y = 0; y < HEIGHT; y++)
            {
                for (x = 0; x < WIDTH-1; x++)
                {
                    sBuf3[y*WIDTH + x] = h_frame_out[y*WIDTH + x];
                }
            }
            sMut3.unlock();
        }
        sHead = (++sHead) % NUM_BUFFERS;
        sCount++;
        
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
        fps = 1000000/duration;
        
        //std::cout << fps << std::endl;
        
        //----------------TCP Send------------------------//
        //TCPSend();
        frame++;
        
        //------------------------------------------------//
    }
    close(receiveSocket);
    close(sendSocket);
    return 0;
}
