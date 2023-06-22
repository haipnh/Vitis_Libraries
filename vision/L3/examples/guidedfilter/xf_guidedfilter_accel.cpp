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

#include "xf_darkchannel_config.h"

extern "C" {

void guided_filter(ap_uint<PTR_IN_WIDTH>* img_in_b,
                  ap_uint<PTR_IN_WIDTH>* img_in_g,
                  ap_uint<PTR_IN_WIDTH>* img_in_r,
                  unsigned char* process_shape,
                  ap_uint<PTR_OUT_WIDTH>* img_out,
                  int rows,
                  int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi      port=img_in_b      offset=slave  bundle=gmem0
    #pragma HLS INTERFACE m_axi      port=img_in_g      offset=slave  bundle=gmem1
    #pragma HLS INTERFACE m_axi      port=img_in_r      offset=slave  bundle=gmem2

    #pragma HLS INTERFACE s_axilite  port=rows 			      
    #pragma HLS INTERFACE s_axilite  port=cols 			      
    #pragma HLS INTERFACE m_axi      port=process_shape offset=slave  bundle=gmem3
    #pragma HLS INTERFACE s_axilite  port=process_shape			      
    #pragma HLS INTERFACE m_axi      port=img_out       offset=slave  bundle=gmem4

    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, XF_NPPC1> imgInput_B(rows, cols);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, XF_NPPC1> imgInput_G(rows, cols);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, XF_NPPC1> imgInput_R(rows, cols);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, XF_NPPC1> imgCache_1(rows, cols);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, XF_NPPC1> imgCache_2(rows, cols);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, XF_NPPC1> imgOutput(rows, cols);

    // Copy the shape data:
    unsigned char _kernel[FILTER_SIZE * FILTER_SIZE];
    for (unsigned int i = 0; i < FILTER_SIZE * FILTER_SIZE; ++i) {
// clang-format off
        #pragma HLS PIPELINE
        // clang-format on
        _kernel[i] = process_shape[i];
    }

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    // Retrieve xf::cv::Mat objects from img_in data:
    xf::cv::Array2xfMat<PTR_IN_WIDTH, XF_8UC1, HEIGHT, WIDTH, XF_NPPC1>(img_in_b, imgInput_B);
    xf::cv::Array2xfMat<PTR_IN_WIDTH, XF_8UC1, HEIGHT, WIDTH, XF_NPPC1>(img_in_g, imgInput_G);
    xf::cv::Array2xfMat<PTR_IN_WIDTH, XF_8UC1, HEIGHT, WIDTH, XF_NPPC1>(img_in_r, imgInput_R);
    
    xf::cv::min<XF_8UC1, HEIGHT, WIDTH, XF_NPPC1>(imgInput_B, imgInput_G, imgCache_1);
    xf::cv::min<XF_8UC1, HEIGHT, WIDTH, XF_NPPC1>(imgCache_1, imgInput_R, imgCache_2);

    xf::cv::erode<XF_BORDER_CONSTANT, XF_8UC1, HEIGHT, WIDTH, XF_KERNEL_SHAPE, FILTER_SIZE, FILTER_SIZE, ITERATIONS,
                  XF_NPPC1>(imgCache_2, imgOutput, _kernel);

    // Convert _dst xf::cv::Mat object to output array:
    xf::cv::xfMat2Array<PTR_OUT_WIDTH, XF_8UC1, HEIGHT, WIDTH, XF_NPPC1>(imgOutput, img_out);

    return;

} // End of kernel
} // End of extern C
