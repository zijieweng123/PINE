from copy import deepcopy

from pyiqa.utils.registry import METRIC_REGISTRY
from .correlation_coefficient import calculate_srcc, calculate_plcc, calculate_krcc, calculate_rmse

__all__ = [
    'calculate_srcc',
    'calculate_plcc',
    'calculate_krcc',
    'calculate_rmse',
]


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    # ``better`` controls best-result tracking in BaseModel; it is metadata,
    # not an argument accepted by metric calculation functions.
    opt.pop('better', None)
    metric = METRIC_REGISTRY.get(metric_type)(*data, **opt)
    return metric
