import logging
import inspect
import pprint

import tectosaur
import tectosaur_topo

from tectosaur.check_for_problems import check_for_problems

def setup_cfg(defaults, kwargs):
    cfg = dict()
    for k, v in defaults.items():
        if k in kwargs:
            cfg[k] = kwargs[k]
        else:
            cfg[k] = defaults[k]
    fnc_name = inspect.stack()[1][3]
    tectosaur_topo.logger.info(str(fnc_name) + ' configuration: \n' + pprint.pformat(cfg))
    tectosaur_topo.cfg.set_logging_levels(cfg['log_level'])
    return cfg

def set_logging_levels(log_level):
    tectosaur_topo.logger.setLevel(log_level)
    tectosaur.logger.setLevel(log_level)

def alert_mesh_problems(m):
    intersections, slivers, short_tris, sharp_angles = check_for_problems((m.pts, m.tris))
    if len(slivers) != 0 or len(short_tris) != 0:
        raise Exception('There are sliver elements in the meshes provided: ' + str((sliver + short_tris)))
    if len(sharp_angles) != 0:
        raise Exception('There are very sharp angles between adjacent elements in the meshes provided: ' + str(sharp_angles))
    if len(intersections) != 0:
        raise Exception('There are intersecting elements in the meshes provided: ' + str(intersections))

