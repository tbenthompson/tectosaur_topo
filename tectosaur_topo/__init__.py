from tectosaur.util.logging import setup_root_logger
logger = setup_root_logger(__name__)

from tectosaur_topo.assemble import forward_assemble
from tectosaur_topo.interior import interior
from tectosaur_topo.solve import forward_solve

def forward(surf, fault, fault_slip, sm, pr, **kwargs):
    system = forward_assemble(surf, fault, sm, pr, **kwargs)
    return forward_solve(system, fault_slip, **kwargs)
