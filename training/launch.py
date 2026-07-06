# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer
import torch
import os
import sys

# Ensure project root (which contains the `training` package) is importable even
# when this script is executed via a filesystem path such as `python training/launch.py`.
# 0, 1 6000 prp
# 3, 2 6000
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def main():
    parser = argparse.ArgumentParser(description="Train model with configurable YAML file")
    parser.add_argument(
        "--config", 
        type=str, 
        default="default_smpl",
        help="Name of the config file (without .yaml extension, default: default)" 
    )
    args = parser.parse_args()

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=args.config)

    print("torch:", torch.__version__)
    print("cuda in torch:", torch.version.cuda)
    print("cuda is available:", torch.cuda.is_available())
    print("device name:", torch.cuda.get_device_name(0))
    print("device capability:", torch.cuda.get_device_capability(0))

    trainer = Trainer(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()


