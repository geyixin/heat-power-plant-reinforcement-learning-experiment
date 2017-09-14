import copy
import itertools
import os
import random
import csv

import keras.models
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam
import pickle


import numpy as np
import pandas as pd


def dump_pickle(obj, name):
    with open(name, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(name):
    with open(name, 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def dense_q(input_length):

    model = Sequential()
    model.add(Dense(500, input_shape=(input_length,), activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(250, input_shape=(input_length,), activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(100, input_shape=(input_length,), activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='linear', kernel_initializer='he_uniform'))
    model.compile(loss='mse', optimizer=adam())

    return model


#  agent based on Double Q-Learning
class DoubleQLearner(object):
    def __init__(self, env, verbose, device, save_csv=True, run_name='TEST'):
        self.env = env
        self.verbose = verbose
        self.path = './results/'+run_name+'/'

        input_length = len(self.env.s_mins) + len(self.env.a_mins)
        self.Q = self.get_value_functions(input_length, device)
        self.batch_size = 64  # size of batch for sampling memory
        self.epochs = 50
        self.memory, self.network_memory, self.info, self.age = self.get_memory()
        self.save_csv = save_csv
        csv_file = open('results.csv', 'wb')
        self.csv_writer = csv.writer(csv_file, delimiter=',', quotechar='|')

        self.epsilon = 0.1
        self.epsilon_decay = 0.9999  # decayed per policy decision
        self.policy_ = 0   # 0 = naive, 1 = e-greedy
        self.discount = 0.9  # discount factor for next_state
        self.test_state_actions = self.get_test_state_actions()

    def get_memory(self, default_epsilon=1):
        paths = ['memory.pickle',
                 'network_memory.pickle',
                 'info.pickle']
        objs = []
        for path in paths:
            if os.path.exists(self.path+path):
                print(path+' already exists')
                objs.append(load_pickle(self.path+path))
                self.age = objs[0][-1][10]
            else:
                print('Creating new memory for '+path)
                self.age = 0
                objs.append([])
        print objs
        if self.age == 0:
            self.hists = [[],[],[]]
            self.epsilon = default_epsilon
        else:
            self.hists = objs[2][-1][3]
            self.epsilon = objs[0][-1][7]

        print('Age is {}.'.format(self.age))
        return objs[0], objs[1], objs[2], self.age

    def get_value_functions(self, input_length, device, value_functions=2):
        mdl_paths = [self.path + 'Q'+str(i)+'.h5' for i in range(value_functions)]
        Q = [None] * value_functions

        for j, path in enumerate(mdl_paths):
            if os.path.exists(path):
                print('Q{} function already exists'.format(j+1))
                Q[j] = keras.models.load_model(path)
            else:
                print('Q{} function being created'.format(j+1))
                Q[j] = dense_q(input_length)
        return Q

    def save_agent_brain(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.Q[0].save(filepath=self.path+'Q1.h5')
        self.Q[1].save(filepath=self.path+'Q2.h5')
        print('saved Q functions')
        dump_pickle(self.memory, self.path+'memory.pickle')
        dump_pickle(self.network_memory, self.path+'network_memory.pickle')
        dump_pickle(self.info, self.path+'info.pickle')
        print('saved memories')

    def single_episode(self, episode_number):
        print('Starting episode ' + str(episode_number))
        done = False
        self.age += 1
        self.env.reset()
        while done is not True:
            state = self.env.state
            action, state_action, choice = self.policy(state)
            next_state, reward, done, env_info = self.env.step(action)
            if self.save_csv:
                self.csv_writer.writerow(state_action)
            # training on non-NAIVE episodes
            if self.policy_ > 0:
                self.hists = self.train_model()

            self.memory.append([copy.copy(state),
                                copy.copy(action),
                                copy.copy(state_action),
                                copy.copy(reward),
                                copy.copy(next_state),
                                copy.copy(episode_number),
                                copy.copy(self.env.steps),
                                copy.copy(self.epsilon),
                                copy.copy(choice),
                                copy.copy(done),
                                copy.copy(self.age)])

            self.network_memory.append([copy.copy(self.normalize([state_action])),
                                        copy.copy(reward),
                                        copy.copy(self.state_to_state_actions(next_state)[1]),
                                        copy.copy(done)])

            self.info.append([
                np.mean(self.Q[0].predict(self.test_state_actions)),
                np.mean(self.Q[1].predict(self.test_state_actions)),
                copy.copy(self.hists)])

            if self.verbose > 0:
                print('episode {} - step {} - choice {}'.format(episode_number, self.env.steps, choice))
                print('epsilon is {}'.format(self.epsilon))
                print('age is {}'.format(self.age))

        self.save_agent_brain()
        print('Finished episode {}.'.format(episode_number))
        print('Age is {}.'.format(self.age))
        return self

    def policy(self, state):
        state = state.values

        if self.policy_ == 0:  # naive
            choice = 'NAIVE'
            action = [action_space.high for action_space in self.env.action_space]

        else:  # self.policy_ == 1 - e-greedy
            if random.random() < self.epsilon:  # exploring
                choice = 'RANDOM'
                action = [np.random.choice(np.array(action_space.sample()).flatten())
                          for action_space in self.env.action_space]
            else:  # acting according to Q1 & Q2
                choice = 'GREEDY'
                # we now use both of our Qfctns to select an action
                state_actions, norm_state_actions = self.state_to_state_actions(state)  # array of shape (num state actions, state_action dim)
                both_returns = [Qfctn.predict(norm_state_actions) for Qfctn in self.Q] # uses both Q fctns
                returns = np.add(both_returns[0], both_returns[1]) / 2

                idx = np.argmax(returns)
                optimal_state_action = state_actions[idx]
                optimal_action = optimal_state_action[len(self.env.state):]

                both_estimates = [rtn[idx] for rtn in both_returns]
                print('Q1 estimate={} - Q2 estimate={}.'.format(both_estimates[0], both_estimates[1]))

                action = optimal_action

        # decaying epsilon
        self.epsilon = self.decay_epsilon()

        action = np.array(action).reshape(-1)
        state_action = np.concatenate([state, action])
        return action, state_action, choice

    def state_to_state_actions(self, state):
        action_space = self.env.create_action_space()
        bounds = [np.arange(asset.low, asset.high + 1) for asset in action_space]
        actions = [np.array(tup) for tup in list(itertools.product(*bounds))]
        state_actions = [np.concatenate((state, a)) for a in actions]
        norm_state_actions = np.vstack(self.normalize(state_actions))
        return state_actions, norm_state_actions

    def normalize(self, state_actions):
        mins, maxs = list(self.env.mins), list(self.env.maxs)
        norm_state_action, norm_state_actions = [], []
        length = 0
        for state_action in state_actions:
            length = len(state_action)
            for j, variable in enumerate(state_action):
                lb, ub = mins[j], maxs[j]
                normalized = (variable - lb) / (ub - lb)
                norm_state_action.append(normalized)

            norm_array = np.array(norm_state_action).reshape(-1, length)
            norm_state_actions.append(norm_array)
            norm_state_action = []
        norm_state_actions = np.array(norm_state_actions).reshape(-1, length)
        return norm_state_actions

    def train_model(self, memory_length=50000):
        if self.verbose > 0:
            print('Starting training')

        # setting our Q functions
        reverse = random.choice([True, False])
        if reverse:  # randomly swapping which Q we use
            Q1 = self.Q[1]
            Q2 = self.Q[0]
            print('training model Q[0] using prediction from Q[1]')
        else:
            Q1 = self.Q[0]
            Q2 = self.Q[1]
            print('training model Q[1] using prediction from Q[0]')

        sample_size = min(len(self.network_memory), self.batch_size)
        memory = random.sample(self.network_memory[-memory_length:], sample_size)
        batch = np.array(memory)

        X = np.hstack(batch[:, 0]).reshape(sample_size, -1)  # state_actions
        reward = batch[:, 1]
        next_state_actions = batch[:, 2]

        # taking advantage of constant action space size here
        num_state_actions = batch.shape[0] * next_state_actions[0].shape[0]
        unstacked = np.vstack(next_state_actions).reshape(num_state_actions, -1)  # shape = (num_state_actions, state_action_length)
        predictions = self.discount * Q1.predict(unstacked)  # shape = (num_state_actions, 1)
        predictions = predictions.reshape(sample_size, # shape = (batch_size,
                                          next_state_actions[0].shape[0], # number of state actions
                                          -1)  # predicted Q1(s,a)

        maximum_returns = np.amax(predictions, 1).reshape(-1)  # shape = (max[Q1(s,a)],)
        Y = np.add(reward, maximum_returns)

        if self.verbose > 0:
            print('Fitting model')

        # fiting model Q2 using predictions from Q1
        hist = Q2.fit(X, Y, epochs=self.epochs, batch_size=sample_size, verbose=0)

        if reverse:
            self.Q[0] = Q2
            self.Q[1] = Q1
            self.hists[0] += hist.history['loss']
            self.hists[1] += [0] * self.epochs
        else:
            self.Q[0] = Q1
            self.Q[1] = Q2
            self.hists[0] += [0] * self.epochs
            self.hists[1] += hist.history['loss']

        self.hists[2] += hist.history['loss']

        return self.hists

    def get_test_state_actions(self):
        Q_test = pd.read_csv('env/chp/Q_test.csv', index_col=[0])
        Q_test.iloc[:, 1:] = Q_test.iloc[:, 1:].apply(pd.to_numeric)
        test_state_actions = np.array(Q_test.iloc[:, 1:])
        test_state_actions = self.normalize(test_state_actions)
        return test_state_actions

    def decay_epsilon(self):
        return self.epsilon * self.epsilon_decay if self.epsilon != 0 else 0

    def create_outputs(self):
        print('Generating outputs')
        memory = pd.DataFrame(self.memory,
                              columns=['State', 'Action', 'State Action',
                                       'Reward', 'Next State', 'Episode',
                                       'Step', 'Epsilon', 'Choice', 'Done', 'Age'])
        return memory

