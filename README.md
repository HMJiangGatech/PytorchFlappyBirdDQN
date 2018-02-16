# PytorchFlappyBirdDQN
DQN for playing Flappy Bird implemented in Pytorch


## Installation Dependencies:
- Python 3
- Pytorch
- pygame
- OpenCV-Python

## Usage
```bash
python dqn_pytorch.py
```

## Training Parameter
You can use the following parameters to train DQN
```pythn
OBSERVE = 10000. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.2 # starting value of epsilon
```

## Disclaimer
This repo is based on the (tensorflow version)[https://github.com/yenchenlin/DeepLearningFlappyBird]
