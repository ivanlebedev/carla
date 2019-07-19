from . import bounding_box_utils, constants, segmentation_utils


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__