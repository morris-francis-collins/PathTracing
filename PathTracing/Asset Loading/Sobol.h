//
//  Sobol.h
//  PathTracing
//
//  Created on 4/19/26.
//

#pragma once

#include <stdint.h>
#include <cstdio>
#include "sobol_data.h"

void fillSobolBuffer(uint32_t* output, int dimensions);
