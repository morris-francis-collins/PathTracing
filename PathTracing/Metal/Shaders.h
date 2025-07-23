//
//  definitions.h
//  PathTracing
//
//  Created on 3/21/25.
//

#pragma once

#include <simd/simd.h>
#include "Utility.h"
#include "Lights.h"
#include "Interactions.h"
#include "Integrators.h"
#include "Materials.h"

#define PIXEL_WIDTH 800.0f
#define PIXEL_HEIGHT 600.0f
#define ASPECT_RATIO (PIXEL_WIDTH / PIXEL_HEIGHT)
#define A 4 * ASPECT_RATIO * pow(tan(M_PI_F * CAMERA_FOV_ANGLE * 0.5f / 180.0f), 2.0f)
