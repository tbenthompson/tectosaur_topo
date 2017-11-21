import logging

import tectosaur
import tectosaur_topo

from tectosaur.check_for_problems import check_for_problems

def check_valid_options(kwargs, cfg):
    for k in kwargs:
        if k not in cfg:
            raise Exception(k + ' is not a valid config option')

def set_logging_levels(verbose):
    if not verbose:
        tectosaur_topo.logger.setLevel(logging.WARNING)
        tectosaur.logger.setLevel(logging.WARNING)
    else:
        tectosaur_topo.logger.setLevel(logging.DEBUG)
        tectosaur.logger.setLevel(logging.DEBUG)

def alert_mesh_problems(m):
    intersections, slivers, short_tris, sharp_angles = check_for_problems((m.pts, m.tris))
    if len(slivers) != 0 or len(short_tris) != 0:
        raise Exception('There are sliver elements in the meshes provided: ' + str((sliver + short_tris)))
    if len(sharp_angles) != 0:
        raise Exception('There are very sharp angles between adjacent elements in the meshes provided: ' + str(sharp_angles))
    if len(intersections) != 0:
        raise Exception('There are intersecting elements in the meshes provided: ' + str(intersections))

