// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_BORISSOLENOID_H
#define XTRACK_BORISSOLENOID_H

#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/elements_src/track_borissolenoid.h"

GPUFUN
void BorisSolenoid_track_local_particle(
    BorisSolenoidData el,
    LocalParticle* part0
) {
    const double L_coil = BorisSolenoidData_get_L_coil(el);
    const double a = BorisSolenoidData_get_a(el);
    const double B0 = BorisSolenoidData_get_B0(el);
    const double z0 = BorisSolenoidData_get_z0(el);
    const double length = BorisSolenoidData_get_length(el);
    const int n_steps = BorisSolenoidData_get_n_steps(el);
    const double shift_x = BorisSolenoidData_get_shift_x(el);
    const double shift_y = BorisSolenoidData_get_shift_y(el);

    if (n_steps <= 0 || length <= 0.0) {
        return;
    }

    START_PER_PARTICLE_BLOCK(part0, part);
        BorisSolenoid_single_particle(
            part,
            L_coil,
            a,
            B0,
            z0,
            length,
            n_steps,
            shift_x,
            shift_y
        );
    END_PER_PARTICLE_BLOCK;
}

#endif // XTRACK_BORISSOLENOID_H
