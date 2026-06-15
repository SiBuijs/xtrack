// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //
#ifndef XTRACK_TRACK_SPLINEBORIS_H
#define XTRACK_TRACK_SPLINEBORIS_H

#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/elements_src/track_boris_integrator.h"

GPUFUN
void SplineBoris_single_particle(
    LocalParticle* part,
    const double  bs[5],
    const double* const *by,
    const double* const *bx,
    const int      multipole_order,
    const double   length,
    const int      n_steps,
    const double   shift_x,
    const double   shift_y,
    const double   scale_b,
    const int64_t  order,
    const double   inv_factorial_order,
    GPUGLMEM const double* knl,
    GPUGLMEM const double* ksl,
    const int64_t  radiation_flag,
    SynchrotronRadiationRecordData radiation_record
){
    boris_track_single_particle(
        part,
        BORIS_FIELD_SPLINE,
        length,
        n_steps,
        bs,
        by,
        bx,
        multipole_order,
        shift_x,
        shift_y,
        scale_b,
        order,
        inv_factorial_order,
        knl,
        ksl,
        radiation_flag,
        radiation_record,
        0.0,
        0.0,
        0.0
    );
}

#endif // XTRACK_TRACK_SPLINEBORIS_H
