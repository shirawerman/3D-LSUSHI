import glob

from Utils import prepare_result_dir
import Config
import DUULM
import torch #noqa
import sys
import json
import os
from datetime import datetime


def update_cp_path(conf):
    mode_cp_path = os.path.join(conf.result_path, "model_cp")
    all_models = glob.glob(f"{mode_cp_path}/*.pth")
    if len(all_models) > 0:
        def my_sort_key(x):
            y = x.split(".")[-2].split("/")[-1]
            y = y.replace("model_cp_", "")
            return float(y)
        all_models = sorted(all_models, key=my_sort_key)
        conf.model_path = all_models[-1]
    return conf


def main(config_path, train="train"):
    with open(config_path) as f:
        conf = Config.Config(json.load(f))
        conf.result_path = prepare_result_dir(conf, config_path)

        # reload model if exists
        if len(conf.model_path) == 0 or train == "test":
            conf = update_cp_path(conf)

        # Train LSUSHI
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Start Time =", current_time)

        exp = DUULM.DUULM(conf)

        now = datetime.now()

        if train == "train":
            exp.run()
        elif train == "test":
            exp.test(conf.model_path)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("done training time =", current_time)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    if len(sys.argv) == 2:
        main(sys.argv[1])
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])

