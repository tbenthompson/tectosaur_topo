import numpy as np
import pprint

from tectosaur.interior import interior_integral
import tectosaur_topo.cfg

import logging
logger = logging.getLogger(__name__)

def interior(obs_pts, m, soln, sm, pr, **kwargs):
    defaults = dict(
        quad_far_order = 3,
        quad_near_order = 8,
        float_type = np.float32,
        log_level = logging.DEBUG
    )
    cfg = tectosaur_topo.cfg.setup_cfg(defaults, kwargs)

    return -interior_integral(
        obs_pts, obs_pts, m, soln, 'elasticT3',
        cfg['quad_far_order'], cfg['quad_near_order'], [sm, pr], cfg['float_type'],
        # fmm_params = [100, 3.0, 3000, 25]
    )
