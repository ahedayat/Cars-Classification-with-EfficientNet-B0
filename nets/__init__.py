"""
Supported Networks:
    - EmbeddingNetwork -> for embedding an image
"""

from .my_efficient_net_b import MyEfficientNetB
from .net_utils import save_net as save
from .net_utils import load_net as load
from .net_utils import schedule_lr

__version__ = '1.0.0'
__author__ = 'Ali Hedayatnia, M.Sc. Student of Artificial Intelligence @ University of Tehran'
