from os.path import join
import logging
import socket
import time
import torch
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig

from GPR_MLIP.data import load_from_npz

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config")
def func(cfg: DictConfig):
    hydra_cfg = HydraConfig.get()
    if hydra_cfg.mode == hydra.types.RunMode.MULTIRUN:
        result_dir = join(hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir)
    elif hydra_cfg.mode == hydra.types.RunMode.RUN:
        result_dir = hydra_cfg.run.dir

    log.info("Running on host: " + str(socket.gethostname()))

    torch.set_default_tensor_type(torch.DoubleTensor)

    # load data
    inputs, targets = hydra.utils.instantiate(cfg.dataset.load_data_function)

    # prepare data
    inds, offset = hydra.utils.instantiate(cfg.prepare_data.function, targets=targets)

    # experiment
    log.info("Run experiment")
    start_time = time.time()
    hydra.utils.instantiate(
        cfg.experiment.function,
        inputs=inputs,
        targets=targets, 
        inds=inds, 
        offset=offset, 
        result_dir=result_dir
    )
    log.info(f'Experiment run time: {time.time() - start_time}')


if __name__ == "__main__":
    func()
