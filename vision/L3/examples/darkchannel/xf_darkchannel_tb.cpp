/*
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common/xf_headers.hpp"
#include "xcl2.hpp"
#include "xf_darkchannel_config.h"
// OpenCV reference function:
void darkchannel(std::vector<cv::Mat>& src, cv::Mat& dst, cv::Mat& kernel) {
    // Temporary matrices for processing
    cv::Mat _mat1, _mat2;
    //std::cout << "DEBUG: darkchannel::Checkpoint 1." << std::endl;
    cv::min(src[0], src[1], _mat1);
    //std::cout << "DEBUG: darkchannel::Checkpoint 2." << std::endl;
    cv::min(_mat1,  src[2], _mat2);
    //std::cout << "DEBUG: darkchannel::Checkpoint 3." << std::endl;
    cv::erode(_mat2, dst, kernel);
    //std::cout << "DEBUG: darkchannel::Checkpoint 4." << std::endl;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <INPUT IMAGE PATH 1>\n", argv[0]);
        return EXIT_FAILURE;
    }

    cv::Mat in_img, out_img, ocv_ref, diff;

    // Open input image:
    in_img = cv::imread(argv[1], 1);
    if (!in_img.data) {
        fprintf(stderr, "ERROR: Could not open the input image.\n ");
        return -1;
    }

    std::vector<cv::Mat> in_bgr;
    cv::split(in_img, in_bgr);

    // Allocate the memory for output images:
    out_img.create(in_img.rows, in_img.cols, CV_8UC1);
    ocv_ref.create(in_img.rows, in_img.cols, CV_8UC1);

    // Get processing kernel with desired shape - used in erode:

    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(FILTER_SIZE, FILTER_SIZE), cv::Point(-1, -1));

    // Create vectors holding thresholds and shape:
    std::vector<unsigned char, aligned_allocator<unsigned char> > shape(FILTER_SIZE * FILTER_SIZE);

    for (int i = 0; i < (FILTER_SIZE * FILTER_SIZE); i++) {
        shape[i] = element.data[i];
    }

    int rows = in_img.rows;
    int cols = in_img.cols;

    std::cout << "INFO: Params prepared." << std::endl;

    // Start time for latency calculation of CPU function

    struct timespec begin_cpu, end_cpu; //, begin_hw, end_hw;
    clock_gettime(CLOCK_REALTIME, &begin_cpu);

    std::cout << "DEBUG: Checkpoint 1." << std::endl;
    // Reference function:
    darkchannel(in_bgr, ocv_ref, element);
    std::cout << "DEBUG: Checkpoint 2." << std::endl;

    // End time for latency calculation of CPU function

    clock_gettime(CLOCK_REALTIME, &end_cpu);
    long seconds, nanoseconds;
    double cpu_time;

    seconds = end_cpu.tv_sec - begin_cpu.tv_sec;
    nanoseconds = end_cpu.tv_nsec - begin_cpu.tv_nsec;
    cpu_time = seconds + nanoseconds * 1e-9;
    cpu_time = cpu_time * 1e3;

    // Write down reference and input image:
    cv::imwrite("outputref.png", ocv_ref);

    // OpenCL section:
    size_t image_in_size  = in_img.rows * in_img.cols * sizeof(unsigned char);
    size_t image_out_size = in_img.rows * in_img.cols * sizeof(unsigned char);
    size_t vector_size = FILTER_SIZE * FILTER_SIZE * sizeof(unsigned char);

    cl_int err;
    std::cout << "INFO: Running OpenCL section." << std::endl;

    // Get the device:
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Contex, command queue and device name:
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
    OCL_CHECK(err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));

    std::cout << "INFO: Device found - " << device_name << std::endl;

    // Load binary:
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_darkchannel");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "dark_channel", &err));

    // Allocate the buffers:
    OCL_CHECK(err, cl::Buffer buffer_inImage_B(context, CL_MEM_READ_ONLY, image_in_size, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_inImage_G(context, CL_MEM_READ_ONLY, image_in_size, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_inImage_R(context, CL_MEM_READ_ONLY, image_in_size, NULL, &err));

    OCL_CHECK(err, cl::Buffer buffer_shapeKrnl(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, vector_size,
                                               shape.data(), &err));
    // printf("finished shape allocation\n");
    OCL_CHECK(err, cl::Buffer buffer_outImage(context, CL_MEM_WRITE_ONLY, image_out_size, NULL, &err));

    // Set kernel arguments:
    OCL_CHECK(err, err = kernel.setArg(0, buffer_inImage_B));
    OCL_CHECK(err, err = kernel.setArg(1, buffer_inImage_G));
    OCL_CHECK(err, err = kernel.setArg(2, buffer_inImage_R));
    OCL_CHECK(err, err = kernel.setArg(3, buffer_shapeKrnl));
    OCL_CHECK(err, err = kernel.setArg(4, buffer_outImage));
    OCL_CHECK(err, err = kernel.setArg(5, rows));
    OCL_CHECK(err, err = kernel.setArg(6, cols));
    // printf("finished set arguments\n");

    // Intialize the buffers:
    cl::Event event;

    OCL_CHECK(err, 
              queue.enqueueWriteBuffer(buffer_inImage_B,    // buffer on the FPGA
                                       CL_TRUE,             // blocking call
                                       0,                   // buffer offset in bytes
                                       image_in_size,       // Size in bytes
                                       in_bgr[0].data       // Pointer to the data to copy
                                       ));

    OCL_CHECK(err, 
              queue.enqueueWriteBuffer(buffer_inImage_G,    // buffer on the FPGA
                                       CL_TRUE,             // blocking call
                                       0,                   // buffer offset in bytes
                                       image_in_size,       // Size in bytes
                                       in_bgr[1].data       // Pointer to the data to copy
                                       ));

    OCL_CHECK(err, 
              queue.enqueueWriteBuffer(buffer_inImage_R,    // buffer on the FPGA
                                       CL_TRUE,             // blocking call
                                       0,                   // buffer offset in bytes
                                       image_in_size,       // Size in bytes
                                       in_bgr[2].data       // Pointer to the data to copy
                                       ));

    // Copy input data to device global memory
    OCL_CHECK(err, err = queue.enqueueMigrateMemObjects({buffer_shapeKrnl}, 0));

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;

    // Execute the kernel:
    OCL_CHECK(err, err = queue.enqueueTask(kernel, NULL, &event));
    clWaitForEvents(1, (const cl_event*)&event);
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    // Copy Result from Device Global Memory to Host Local Memory
    queue.enqueueReadBuffer(buffer_outImage, // This buffers data will be read
                            CL_TRUE,         // blocking call
                            0,               // offset
                            image_out_size,
                            out_img.data // Data will be stored here
                            );

    // Clean up:
    queue.finish();

    // Write down the kernel output image:
    cv::imwrite("output.png", out_img);

    // Results verification:
    int cnt = 0;
    cv::absdiff(ocv_ref, out_img, diff);
    cv::imwrite("diff.png", diff);

    for (int i = 0; i < diff.rows; ++i) {
        for (int j = 0; j < diff.cols; ++j) {
            uchar v = diff.at<uchar>(i, j);
            if (v > 0) cnt++;
        }
    }

    float err_per = 100.0 * (float)cnt / (diff.rows * diff.cols);

    std::cout << "INFO: Verification results:" << std::endl;
    std::cout << "\tPercentage of pixels above error threshold = " << err_per << "%" << std::endl;

    if (err_per > 0.0f) {
        fprintf(stderr, "ERROR: Test Failed.\n ");
        return EXIT_FAILURE;
    }

    std::cout.precision(3);
    std::cout << std::fixed;

    std::cout << "Latency for CPU function is: " << cpu_time << "ms" << std::endl;

    return 0;
}
