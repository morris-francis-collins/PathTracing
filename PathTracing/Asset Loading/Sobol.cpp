//
//  Sobol.cpp
//  PathTracing
//
//  Created on 4/19/26.
//

#include "Sobol.h"

void fillSobolBuffer(uint32_t* output, int dimensions) {
    if (dimensions > sobol_data::kMaxSobolDim) {
        printf("Sobol dimensions exceeds max dimensions");
        dimensions = sobol_data::kMaxSobolDim;
    }
    
    for (int d = 0; d < dimensions; d++) {
        for (int k = 0; k < 31; k++) {
            output[d * 32 + k] = static_cast<uint32_t>(sobol_data::kDirectionNumbers[d][k]) << (31u - k);
        }
        output[d * 32 + 31] = 0;
    }
}
