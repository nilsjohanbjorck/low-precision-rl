# Low-Precision Reinforcement Learning: Running Soft Actor-Critic in Half Precision -- ICML 2021

### Johan Bjorck, Xiangyu Chen, Christopher De Sa, Carla P. Gomes, Kilian Q. Weinberger


## Overview


Low-precision training has become a popular approach to reduce compute requirements, memory footprint, and energy consumption in supervised
learning. In our ICML 2021 paper, we consider training reinforcement learning agents in low precision. Naively training in fp16 does not work well. After six modifications, we demonstrate that low-precision RL trains stably while decreasing computational/memory demands. This codebase contains code for our main experiments. Configuration and command-line arguments are handled via the excellent [hydra framework](https://github.com/facebookresearch/hydra). 


[[paper](http://proceedings.mlr.press/v139/bjorck21a/bjorck21a.pdf)]


## Installation

* You will need an Nvidia GPU with a reasonably recent CUDA version to run the code.

* Create an environment from ```env.yml``` via:

```  
conda env create -f env.yml
conda activate lowprec_rl
```

* Install deepmind control suite as per [here](https://github.com/deepmind/dm_control).

*  You will need to set appropriate environment flags, e.g ```MUJOCO_GL=egl```. You may also consider the flags ```HYDRA_FULL_ERROR=1``` and ```OMP_NUM_THREADS=1```.



## Usage

1. To run an experiment in fp32 on the ```finger_spin ``` environment with seed ```123``` use:
    ```
    python train.py env=finger_spin seed=123
    ```
    Results will appear in a folder named ```runs```.


2. To use half-precision (fp16) for the `actor`, `critic`, and `alpha` use the code below. Note, this is expected to crash.
    ```
    python train.py env=finger_spin seed=123 \
        agent.params.actor_half=True agent.params.crit_half=True agent.params.alpha_half=True
    ```

3. The command above typically crashes without our proposed methods. 
    Our proposed methods can be independently toggled with
    | Method                | Flags                                                            |
    |-----------------------|------------------------------------------------------------------|
    | hAdam                 | `agent.params.use_num_adam=True`                                 |
    | compound loss scaling | `agent.params.use_grad_scaler=True agent.params.adam_eps=0.0001` |
    | normal-fix            | `diag_gaussian_actor.params.stable_normal=True`                  |
    | softplus-fix          | `diag_gaussian_actor.params.tanh_threshold=10`                   |
    | Kahan-momentum        | `agent.params.soft_update_scale=10000`                           |
    | Kahan-gradients       | `agent.params.alpha_kahan=True agent.params.crit_kahan=True`     |



4. To apply all proposed methods,
    ```
    python train.py env=finger_spin seed=123 \
        agent.params.actor_half=True agent.params.crit_half=True agent.params.alpha_half=True \
        agent.params.use_grad_scaler=True agent.params.adam_eps=0.0001 agent.params.use_num_adam=True \
        diag_gaussian_actor.params.tanh_threshold=10 diag_gaussian_actor.params.stable_normal=True \
        agent.params.soft_update_scale=10000 agent.params.alpha_kahan=True agent.params.crit_kahan=True
    ```


  
## Citation

```
@inproceedings{bjorck2021low,
  title={Low-Precision Reinforcement Learning: Running Soft Actor-Critic in Half Precision},
  author={Bj{\"o}rck, Johan and Chen, Xiangyu and De Sa, Christopher and Gomes, Carla P and Weinberger, Kilian},
  booktitle={International Conference on Machine Learning},
  pages={980--991},
  year={2021},
  organization={PMLR}
}

```


## Acknowledgements

The starting point for our codebase is [pytorch_sac](https://github.com/denisyarats/pytorch_sac). 

