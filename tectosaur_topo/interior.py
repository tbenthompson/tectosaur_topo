import numpy as np
import pprint

from tectosaur.interior import interior_integral
import tectosaur_topo.cfg

import logging
logger = logging.getLogger(__name__)

def interior(obs_pts, m, soln, sm, pr, **kwargs):
    cfg = setup_interior_cfg(kwargs)
    tectosaur_topo.cfg.set_logging_levels(cfg['verbose'])
    logger.info('tectosaur_topo.evaluate_interior configuration: \n' + pprint.pformat(cfg))

    return -interior_integral(
        obs_pts, obs_pts, m, soln, 'elasticT3',
        cfg['quad_far_order'], cfg['quad_near_order'], [sm, pr], cfg['float_type'],
        # fmm_params = [100, 3.0, 3000, 25]
    )

def setup_interior_cfg(kwargs):
    cfg = dict(
        quad_far_order = 3,
        quad_near_order = 8,
        float_type = np.float32,
        verbose = True
    )
    tectosaur_topo.cfg.check_valid_options(kwargs, cfg)
    cfg.update(kwargs)
    return cfg
