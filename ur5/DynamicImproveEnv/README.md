
# Normalized Advantage Function (NAF)

PyTorch implementation of the NAF algorithm based on the paper: [Continuous Deep Q-Learning with Model-based Acceleration](https://arxiv.org/abs/1603.00748).

Two versions are implemented: 
1. Jupyter notebook version
2. Script version (results tracking with [wandb](www.wandb.com))

### Recently added PER and n-step method

To run the script version: `python naf.py` 

with the arguments:

    '-env' : Name of the environment (default: Pendulum-v0)
    '-info' : Name of the Experiment (default: Experiment-1)
    '-f', --frames : Number of training frames (default: 40000)   
    '-mem' : Replay buffer size (default: 100000)
    '-b', --batch_size : Batch size (default: 128)
    '-l', --layer_size : Neural Network layer size (default: 256)
    '-g'--gamma : Discount factor gamma (default: 0.99)
    '-t', --tau : Soft update factor tau (default: 1e-3)
    '-lr', --learning_rate : Learning rate (default: 1e-3)
    '-u', --update_every : update the network every x step (default: 1)
    '-n_up', --n_updates : update the network for x steps (default: 1)
    '-s', --seed : random seed (default: 0)
    '-per', choices=[0,1] : Use prioritized experience replay (default: 0)
    '-nstep' : nstep_bootstrapping (default: 1)
    '-d2rl': Using Deep Dense Network if set to 1 (default: 0)
    '--eval_every': Doing an evaluation of the current policy every X frames (default: 1000)
    '--eval_runs': Number of evaluation runs - performance is averaged over all runs (default: 3)



![alttext](/imgs/NAF.png)

In the paper they compared NAF with DDPG and showed faster and more stable learning:  *We show that, in comparison to recently proposed deep actor-critic algorithms, our method tends to learn faster and acquires more accurate policies.*

To verify and support their statement I tested NAF on Pendulum-v0 and LunarLanderConinuous-v2 and compared it with the results of my implementation of [DDPG](https://github.com/BY571/DDPG).

**The results shown do not include the model-based acceleration! Only the base NAF algorithm was tested.**

![alttext](/imgs/NAF_vs_DDPG.png)

![alttext](/imgs/NAF_vs_DDPG_LL_.png)

Indeed the results show a faster and more stable learning!

## TODO:
- Test with Double Q-nets like SAC
- Test with Entropy Regularization (like sac)
- Test with REDQ Q-Net ensemble



Feel free to use this code for your own projects or research:

 ```
@misc{Normalized Advantage Function,
    author = {Dittert, Sebastian},
    title = {PyTorch Implementation of Normalized Advantage Function},
    year = {2020},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/BY571/NAF}},
  }
  ```
