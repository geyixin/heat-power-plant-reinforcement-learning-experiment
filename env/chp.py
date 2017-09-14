# -*- coding: utf-8 -*-
"""
This is a simulation of CHP plant
"""
import gym
import random
from generator import GasEngine
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd


class CombinedHeatPowerPlant(gym.Env):
    def __init__(self, episode_length, lag, random_ts, verbose):
        self.episode_length = episode_length
        self.lag = lag
        self.random_ts = random_ts
        self.verbose = verbose
        self.state_names = []

        self.actual_state, self.visible_state = self.load_data(self.episode_length, self.lag, self.random_ts)

        self.state_models = [
            {'Name': 'Settlement period', 'Min': 0, 'Max': 48},
            {'Name': 'HGH demand', 'Min': 0, 'Max': 30},
            {'Name': 'LGH demand', 'Min': 0, 'Max': 20},
            {'Name': 'Cooling demand', 'Min': 0, 'Max': 10},
            {'Name': 'Electrical demand', 'Min': 0, 'Max': 20},
            {'Name': 'Ambient temperature', 'Min': 0, 'Max': 30},
            {'Name': 'Gas price', 'Min': 15, 'Max': 25},
            {'Name': 'Import electricity price', 'Min': -200, 'Max': 1600},
            {'Name': 'Export electricity price', 'Min': -200, 'Max': 1600}]

        self.asset_models = [
            GasEngine(size=25, name='GT 1'),
            GasEngine(size=25, name='GT 2')
        ]

        self.state = self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.steps = 0
        self.state_df, self.actual_state_df = self.load_data(self.episode_length, self.lag, self.random_ts)
        self.state = self.state_df.iloc[self.steps, 1:]

        self.state_names = [d['Name'] for d in self.state_models]
        self.action_names = [var['Name']
                             for asset in self.asset_models
                             for var in asset.variables]

        self.s_mins, self.s_maxs = self.state_mins_maxs()
        self.a_mins, self.a_maxs = self.asset_mins_maxs()
        self.mins = np.append(self.s_mins, self.a_mins)
        self.maxs = np.append(self.s_maxs, self.a_maxs)

        self.seed()
        self.info = []
        self.done = False
        [asset.reset() for asset in self.asset_models]
        self.last_actions = [var['Current']
                             for asset in self.asset_models
                             for var in asset.variables]

        self.observation_space = self.create_obs_space()
        self.action_space = self.create_action_space()
        return self.state

    def _step(self, actions):
        actual_state = self.actual_state.iloc[self.steps, 1:]
        time_stamp = pd.to_datetime(self.actual_state.iloc[self.steps, 0])

        # taking actions
        for k, asset in enumerate(self.asset_models):
            for var in asset.variables:
                action = actions[k]

                # case at full load
                if var['Current'] == var['Max']:
                    var['Current'] = min([var['Max'],
                                                 var['Current'] + action])

                # case at minimum load
                elif var['Current'] == var['Min']:
                    if (var['Current'] + action) < var['Min']:
                        var['Current'] = 0
                    elif (var['Current'] + action) >= var['Min']:
                        var['Current'] = var['Min'] + action

                # case at off
                elif var['Current'] == 0:
                    if action < 0:
                        var['Current'] = 0
                    else:
                        var['Current'] = var['Min']

                # case in all other times
                else:
                    new = var['Current'] + action
                    new = min(new, var['Max'])
                    new = max(new, var['Min'])
                    var['Current'] = new

                asset.update()

        episode_info = {}
        # sum of energy inputs/outputs for all assets
        total_gas_burned = sum([asset.gas_burnt for asset in self.asset_models])
        total_hgh_gen = sum([asset.HG_heat_output for asset in self.asset_models])
        total_lgh_gen = sum([asset.LG_heat_output for asset in self.asset_models])
        total_cool_gen = sum([asset.cooling_output for asset in self.asset_models])
        total_elect_gen = sum([asset.power_output for asset in self.asset_models])

        episode_info["Total Gas Burned By Turbines"] = total_gas_burned
        episode_info["Total Electricity Generated"] = total_elect_gen
        episode_info["Total LGH Generated"] = total_lgh_gen
        episode_info["Total HGH Generated"] = total_hgh_gen
        episode_info["Total Heat Generated"] = total_lgh_gen + total_hgh_gen


        # energy demands
        elect_dem = actual_state['Electrical']
        hgh_demand = actual_state['HGH']
        lgh_demand = actual_state['LGH']
        cool_demand = actual_state['Cooling']

        episode_info["Electricity Demand"] = elect_dem
        episode_info["HGH Demand"] = hgh_demand
        episode_info["LGH Demand"] = lgh_demand
        total_heat_demand = hgh_demand + lgh_demand
        episode_info["Total Heat Demand"] = total_heat_demand

        # energy balances
        hgh_balance = hgh_demand - total_hgh_gen
        lgh_balance = lgh_demand - total_lgh_gen
        cooling_balance = cool_demand - total_cool_gen

        # backup gas boiler to pick up excess load
        backup_blr = max(0, hgh_balance) + max(0, lgh_balance)
        gas_burned = total_gas_burned + (backup_blr / 0.8)
        episode_info["Boiler Backup"] = backup_blr
        episode_info["Total Gas Burned By Boilers"] = backup_blr / 0.8

        # backup electric chiller for cooling load
        backup_chiller = max(0, cooling_balance)
        backup_chiller_elect = backup_chiller / 3
        elect_dem += backup_chiller_elect

        # electricity balance
        elect_bal = elect_dem - total_elect_gen
        import_elect = max(0, elect_bal)
        export_elect = abs(min(0, elect_bal))

        # all prices in £/MWh
        gas_price = actual_state['Gas price']
        import_price = actual_state['Import electricity price']
        export_price = actual_state['Export electricity price']
        gas_cost = (gas_price * gas_burned) / 2  # £/HH
        import_cost = (import_price * import_elect) / 2  # £/HH
        export_revenue = (export_price * export_elect) / 2  # £/HH

        reward = export_revenue - (gas_cost + import_cost)  # £/HH
        episode_info["Import Cost"] = import_cost
        episode_info["Export Revenue"] = export_revenue
        episode_info["Episode Reward"] = reward

        settlement_period = actual_state['Settlement period']

        #self.info.append([settlement_period,
        #                  total_elect_gen,
        #                  import_price,
        #                  total_heat_demand,
        #                  time_stamp])

        self.steps += int(1)
        if self.steps == (self.episode_length - abs(self.lag) - 1):
            self.done = True

        next_state = self.visible_state.iloc[self.steps, 1:]  # visible state
        self.state = next_state

        self.last_actions = [var['Current']
                             for asset in self.asset_models
                             for var in asset.variables]

        self.action_space = self.create_action_space()

        return next_state, reward, self.done, episode_info

    @staticmethod
    def load_data(episode_length, lag, random_ts):
        ts = pd.read_csv('env/chp/time_series.csv', index_col=[0])
        ts.iloc[:, 1:] = ts.iloc[:, 1:].apply(pd.to_numeric)
        ts.loc[:, 'Timestamp'] = pd.to_datetime(ts.loc[:, 'Timestamp'], dayfirst=True)

        idx = random.randint(0, episode_length) if random_ts else 0

        ts = ts.iloc[idx:idx + episode_length, :]

        if lag < 0:
            actual_state = ts.iloc[:lag, :]
            visible_state = ts.shift(lag).iloc[:lag, :]
        elif lag > 0:
            actual_state = ts.iloc[lag:, :]
            visible_state = ts.shift(lag).iloc[lag:, :]
        else:
            actual_state = ts.iloc[:, :]
            visible_state = ts.iloc[:, :]

        assert actual_state.shape == visible_state.shape

        return visible_state, actual_state

    def create_action_space(self):
        action_space = []
        for j, asset in enumerate(self.asset_models):
            radius = asset.variables[0]['Radius']
            space = gym.spaces.Box(low=-radius,
                                   high=radius,
                                   shape=(1,))
            action_space.append(space)
        return action_space

    def make_outputs(self):
        env_info = pd.DataFrame(self.info,
                                columns=['Settlement period',
                                         'Power generated [MWe]',
                                         'Import electricity price [£/MWh]',
                                         'Total heat demand [MW]',
                                         'Timestamp'])
        env_info.loc[:, 'Timestamp'] = env_info.loc[:, 'Timestamp'].apply(pd.to_datetime)
        env_info.set_index(keys='Timestamp', inplace=True, drop=True)
        return env_info

    def create_obs_space(self):
        states = []
        for mdl in self.state_models:
            states.append([mdl['Min'], mdl['Max']])
            self.state_names.append(mdl['Name'])
        return spaces.MultiDiscrete(states)

    def state_mins_maxs(self):
        s_mins, s_maxs = np.array([]), np.array([])
        for mdl in self.state_models:
            s_mins = np.append(s_mins, mdl['Min'])
            s_maxs = np.append(s_maxs, mdl['Max'])
        return s_mins, s_maxs

    def asset_mins_maxs(self):
        a_mins, a_maxs = [], []
        for j, asset in enumerate(self.asset_models):
            for var in asset.variables:
                a_mins = np.append(a_mins, var['Min'])
                a_maxs = np.append(a_maxs, var['Max'])
        return a_mins, a_maxs

    def asset_states(self):
        asset_states = {}
        for asset in self.asset_models:
            asset_states[asset.name] = []
            for var in asset.variables:
                asset_states[asset.name].append({
                    "Name": var['Name'],
                    "Status": var['Current']
                })
        return asset_states
