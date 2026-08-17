"""P3 (priority-3 paper): sensor-independent continuous scene field.

A single parametric field F(x,y,lambda) is observed through per-sensor linear
operators O_s; a field fitted on sensors A,B is rendered for an unseen sensor C
zero-shot.  This package is the non-neural scaffold (least-squares fitting,
no networks); the learned field and the delta_sensor evaluation live on top.
"""

from . import selfcheck as _selfcheck
from . import field, metrics, sensors
from .field import SceneField
from .sensors import Sensor, fit_field
from .metrics import delta_sensor, relative_error

__all__ = ["SceneField", "Sensor", "fit_field", "delta_sensor",
           "relative_error", "field", "metrics", "sensors"]