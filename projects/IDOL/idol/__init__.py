from .config import add_idol_config
from .idol import IDOL
from .data import YTVISDatasetMapper, build_detection_train_loader, build_detection_test_loader
from .backbone.swin import D2SwinTransformer