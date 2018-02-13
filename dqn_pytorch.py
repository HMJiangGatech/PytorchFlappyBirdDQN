#!/usr/bin/env python
from __future__ import print_function

import cv2
import sys
import os
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

# Gmae Auxilary functions and definition
def get_screen():
    screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))  # transpose into torch order (CHW)

    # Convert to float, rescare, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).type(Tensor)

# Parameter setting
GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.0001 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding = 2)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding = 1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding = 1)
        self.fc1 = nn.Linear(1600, 512)
        self.head = nn.Linear(512, ACTIONS)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.fc1(x.view(x.size(0), -1))
        return self.head(x)

model = DQN()
if use_cuda:
    model.cuda()
optimizer = optim.RMSprop(model.parameters())

# Auxilary functions
def select_action(state,epsilon):
    sample = random.random()
    if sample > epsilon:
        readout_t = model(Variable(state, volatile=True).type(FloatTensor)).data
        return readout_t.max(1)[1].view(1, 1)[0,0]
    else:
        return random.randrange(ACTIONS)

# Optimizer
# def optimize_model():
#     if len(memory) < BATCH_SIZE:
#         return
#     transitions = memory.sample(BATCH_SIZE)
#     # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
#     # detailed explanation).
#     batch = Transition(*zip(*transitions))
#
#     # Compute a mask of non-final states and concatenate the batch elements
#     non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
#                                           batch.next_state)))
#
#     # We don't want to backprop through the expected action values and volatile
#     # will save us on temporarily changing the model parameters'
#     # requires_grad to False!
#     non_final_next_states = Variable(torch.cat([s for s in batch.next_state
#                                                 if s is not None]),
#                                      volatile=True)
#     state_batch = Variable(torch.cat(batch.state))
#     action_batch = Variable(torch.cat(batch.action))
#     reward_batch = Variable(torch.cat(batch.reward))
#
#     # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
#     # columns of actions taken
#     state_action_values = model(state_batch).gather(1, action_batch)
#
#     # Compute V(s_{t+1}) for all next states.
#     next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
#     next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
#     # Now, we don't want to mess up the loss with a volatile flag, so let's
#     # clear it. After this, we'll just end up with a Variable that has
#     # requires_grad=False
#     next_state_values.volatile = False
#     # Compute the expected Q values
#     expected_state_action_values = (next_state_values * GAMMA) + reward_batch
#
#     # Compute Huber loss
#     loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
#
#     # Optimize the model
#     optimizer.zero_grad()
#     loss.backward()
#     for param in model.parameters():
#         param.grad.data.clamp_(-1, 1)
#     optimizer.step()

def playGame():

    # loading networks
    if os.path.isfile("saved_network/checkpointdict.npy") :
        checkpointdict = np.load("saved_network/checkpointdict.npy").item()
        checkpoint = torch.load(checkpointdict["checkpoint_path"])
        model_path = checkpoint["model_path"]
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Successfully loaded:", "saved_network/checkpoint.pth.tar")
    else:
        print("Could not find old network weights")

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)
    s_t = np.ascontiguousarray(s_t, dtype=np.float32)

    # start training
    epsilon = INITIAL_EPSILON
    t = 0

    while "flappy bird" != "angry bird":
        # choose an action epsilon greedily
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            state_tensor = torch.from_numpy(s_t).unsqueeze(0).type(Tensor)
            a_t[select_action(state_tensor,epsilon)] = 1
        else:
            a_t[0] = 1 # do nothing

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (1,80, 80))
        s_t1 = np.append(x_t1, s_t[:3, :, :], axis=0)
        s_t1 = np.ascontiguousarray(s_t1, dtype=np.float32)

        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        # if t > OBSERVE:
        #     # sample a minibatch to train on
        #     minibatch = random.sample(D, BATCH)
        #
        #     # get the batch variables
        #     s_j_batch = [d[0] for d in minibatch]
        #     a_batch = [d[1] for d in minibatch]
        #     r_batch = [d[2] for d in minibatch]
        #     s_j1_batch = [d[3] for d in minibatch]
        #
        #     y_batch = []
        #     readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
        #     for i in range(0, len(minibatch)):
        #         terminal = minibatch[i][4]
        #         # if terminal, only equals reward
        #         if terminal:
        #             y_batch.append(r_batch[i])
        #         else:
        #             y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))
        #
        #     # perform gradient step
        #     train_step.run(feed_dict = {
        #         y : y_batch,
        #         a : a_batch,
        #         s : s_j_batch}
        #     )

        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            save_path = 'saved_networks/' + GAME + '-dqn' + str(t) + ".pth.tar"
            np.save("saved_network/checkpointdict.npy",{'checkpoint_path':save_path})
            torch.save({
                't': t,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict()
            }, save_path)


        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t)
        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''

def main():
    playGame()

if __name__ == "__main__":
    main()
