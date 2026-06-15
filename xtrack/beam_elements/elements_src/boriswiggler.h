// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //

#ifndef XTRACK_BORISWIGGLER_H
#define XTRACK_BORISWIGGLER_H

#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/elements_src/track_boris_integrator.h"


GPUFUN
void BorisWiggler_track_local_particle(BorisWigglerData el, LocalParticle* part0){

    const double length = BorisWigglerData_get_length(el);
    const int n_steps = BorisWigglerData_get_n_steps(el);
    const double k_u = BorisWigglerData_get_k_u(el);
    const double b_tilde = BorisWigglerData_get_b_tilde(el);
    const double s_offset = BorisWigglerData_get_s_offset(el);

    if (n_steps <= 0) {
        return;
    }

    START_PER_PARTICLE_BLOCK(part0, part);
        boris_track_single_particle(
            part,
            BORIS_FIELD_WIGGLER,
            length,
            n_steps,
            NULL,
            NULL,
            NULL,
            0,
            0.0,
            0.0,
            1.0,
            -1,
            1.0,
            NULL,
            NULL,
            0,
            NULL,
            k_u,
            b_tilde,
            s_offset
        );
    END_PER_PARTICLE_BLOCK;
}

#endif /* XTRACK_BORISWIGGLER_H */
