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

#ifndef _XF_DARKCHANNEL_CONFIG_H_
#define _XF_DARKCHANNEL_CONFIG_H_

#include "ap_int.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "hls_stream.h"
#include "core/xf_arithm.hpp"
#include "imgproc/xf_erosion.hpp"
//#include "xf_config_params.h"

/* Set the image height and width */
#define HEIGHT 2160
#define WIDTH 3840

/* Erode and Dilate parameters */
#define FILTER_SIZE 15
#define XF_KERNEL_SHAPE XF_SHAPE_RECT
#define ITERATIONS 1

// Set the input and output pixel depth:
#define IN_TYPE XF_8UC1
#define OUT_TYPE XF_8UC1
#define PTR_IN_WIDTH 8
#define PTR_OUT_WIDTH 8

#endif
