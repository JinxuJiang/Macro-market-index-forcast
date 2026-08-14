"""市场状态模型的数据接入层。"""

from .adapter import CrossSectionalDataAdapter
from .engine import MarketDataEngine, load_config

__all__ = ["CrossSectionalDataAdapter", "MarketDataEngine", "load_config"]

