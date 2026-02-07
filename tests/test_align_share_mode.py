from __future__ import annotations

import numpy as np

from chasecrit.policies import evader_step


def test_align_share_mode_extremes_change_direction() -> None:
    rng = np.random.default_rng(0)

    pos_e = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [0.5, 0.5],
        ],
        dtype=np.float64,
    )
    vel_e = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )
    alive_mask = np.array([True, True, True], dtype=bool)
    pos_p = np.array([[2.0, 0.0]], dtype=np.float64)
    pos_z = np.zeros((0, 2), dtype=np.float64)
    zone_active = np.zeros((0,), dtype=bool)
    world_size = np.array([50.0, 50.0], dtype=np.float64)

    common = dict(
        pos_e=pos_e,
        vel_e=vel_e,
        alive_mask=alive_mask,
        pos_p=pos_p,
        pos_z=pos_z,
        zone_active=zone_active,
        world_size=world_size,
        periodic=False,
        v_e=1.0,
        inertia=1.0,
        r_nbr=2.0,
        r_pred=5.0,
        r_detect=1.0,
        r_sep=0.1,
        w_goal=0.0,
        w_avoid=1.0,
        w_explore=0.0,
        sep_strength=1.0,
        angle_noise=0.0,
        align_control_mode="share",
    )

    v_align_only = evader_step(rng=np.random.default_rng(1), w_align=1.0, **common)
    v_non_align = evader_step(rng=np.random.default_rng(1), w_align=0.0, **common)

    # Focal evader (index 0): alignment points to +x while avoidance points to -x.
    assert v_align_only[0, 0] > 0.0
    assert v_non_align[0, 0] < 0.0
