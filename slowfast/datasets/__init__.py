#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .ava_dataset import Ava  # noqa
from .build import DATASET_REGISTRY, build_dataset  # noqa
from .charades import Charades  # noqa
from .imagenet import Imagenet  # noqa
from .kinetics import Kinetics  # noqa
from .ssv2 import Ssv2  # noqa
from .virat import Virat
from .virat_mask import Viratmask
from .virat_bbn import Viratbbn
from .virat_utils import dict_class_id
try:
    from .ptv_datasets import Ptvcharades, Ptvkinetics, Ptvssv2  # noqa
except Exception:
    print("Please update your PyTorchVideo to latest master")
