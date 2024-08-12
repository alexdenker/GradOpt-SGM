import numpy as np

np.object = object 


import odl
from odl import uniform_discr
from odl.contrib.torch import OperatorModule
from torch import Tensor


class SimpleTrafo():
    def __init__(self, im_shape, num_angles):
        domain = uniform_discr(
            [-im_shape[0] // 2, -im_shape[1] // 2],
            [im_shape[0] // 2, im_shape[1] // 2],
            (im_shape[0], im_shape[1]),
            dtype=np.float32,
        )

        geometry = odl.tomo.parallel_beam_geometry(domain, num_angles=num_angles)
        self._angles = geometry.angles

        ray_trafo_op = odl.tomo.RayTransform(domain, geometry, impl="astra_cuda")
        obs_shape = ray_trafo_op.range.shape
        ray_trafo_op_fun = OperatorModule(ray_trafo_op)
        ray_trafo_adjoint_op_fun = OperatorModule(ray_trafo_op.adjoint)
        fbp_fun = OperatorModule(odl.tomo.fbp_op(ray_trafo_op))

        self.ray_trafo_op_fun = ray_trafo_op_fun
        self.ray_trafo_adjoint_op_fun = ray_trafo_adjoint_op_fun
        self.fbp_fun = fbp_fun

    @property
    def angles(self) -> np.ndarray:
        """:class:`np.ndarray` : The angles (in radian)."""
        return self._angles

    def trafo(self, x: Tensor):
        return self.ray_trafo_op_fun(x)

    def trafo_adjoint(self, x: Tensor):
        return self.ray_trafo_adjoint_op_fun(x)

    def fbp(self, x: Tensor):
        return self.fbp_fun(x)
