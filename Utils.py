import os
import glob
from time import strftime, localtime
from shutil import copy
import scipy.io
import torch
import h5py
import math


def prepare_result_dir(conf, config_path):
    # Create results directory
    if conf.create_results_dir:
        conf.result_path += '/' + conf.exp_name + strftime('_%b_%d_%H_%M_%S', localtime())
        if os.path.isdir(conf.result_path) == False:
            os.makedirs(conf.result_path)
            os.makedirs(conf.result_path+'/model_cp')

        # copy current directory
        local_dir = os.path.dirname(__file__)
        for py_file in glob.glob(local_dir + '/*.py'):
            copy(py_file, conf.result_path)

        # Put a copy of config json
        copy(config_path, conf.result_path)

        # cd to results dir
        os.chdir(conf.result_path)

    return conf.result_path


def norm_tensor(t):
  t_norm = (t-t.min()) / (t.max()-t.min()+1e-10)
  return t_norm


def log_compression(t):
    mx = t.max() + 1e-6
    t = torch.ones_like(t, device=t.device) + 100 * t / mx
    return mx*(torch.log10(t) / torch.log10(torch.tensor([101]).to(t.device)))


def load_data(path):
    if "h5" in path:
        seq = load_h5_file(path)
    else:
        seq = load_mat(path)
    return seq


def load_mat(path):
    mat_dic = scipy.io.loadmat(path)
    seq = mat_dic[list(mat_dic)[-1]]

    return torch.tensor(seq, dtype=torch.float32)


def load_h5_file(path):
    with h5py.File(path, "r+") as mat_dic:
        seq = mat_dic[list(mat_dic)[-1]]
        seq = torch.tensor(seq, dtype=torch.float32).transpose(1, 2)

    return seq
