/*
 * Copyright (C) 2019-2022, Xilinx, Inc.
 * Copyright (C) 2022-2023, Advanced Micro Devices, Inc.
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

#ifndef ADF_GRAPH_H
#define ADF_GRAPH_H

#include <adf.h>

#include "config.h"
#include "kernels.h"

using namespace adf;

/*
 * Cardano dataflow graph to compute weighted moving average of
 * the last 8 samples in a stream of numbers
 */

class thresholdGraph : public adf::graph {
   private:
    kernel k1;

   public:
    port<input> in1;
    port<output> out1;
    port<input> threshVal;
    port<input> maxVal;

    thresholdGraph() {
        // create kernels
        k1 = kernel::create(threshold);
        // create nets to connect kernels and IO ports
        adf::connect<>(in1, k1.in[0]);
        adf::connect<>(k1.out[0], out1);

        adf::dimensions(k1.in[0]) = {ELEM_WITH_METADATA};
        adf::dimensions(k1.out[0]) = {ELEM_WITH_METADATA};

        connect<parameter>(threshVal, async(k1.in[1]));
        connect<parameter>(maxVal, async(k1.in[2]));
        // specify kernel sources
        source(k1) = "xf_threshold.cc";
        // specify kernel run times
        runtime<ratio>(k1) = 0.5;
    }
};

#endif
