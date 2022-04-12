import numpy as np
from casadi.tools import *
import time
import datetime as dt
import os
import pytz
from base_types import Env
from Environments.house_4pumps_helper import HPutils, DataReader
import json
from datetime import datetime


def GetConfig(env_params: dict):
    config = {}
    config['DataPath'] = os.path.abspath(os.path.join(env_params['json_path'], '../../../Data/House4Pumps'))
    config['SensiboDevices'] = ['main', 'living', 'studio', 'livingdown']
    local_timezone = pytz.timezone('Europe/Oslo')

    config['Start'] = dt.datetime(env_params['start'][0],
                                  env_params['start'][1],
                                  env_params['start'][2],
                                  env_params['start'][3],
                                  env_params['start'][4]).astimezone(local_timezone)
    config['Stop'] = dt.datetime(env_params['stop'][0],
                                 env_params['stop'][1],
                                 env_params['stop'][2],
                                 env_params['stop'][3],
                                 env_params['stop'][4]).astimezone(local_timezone)

    config['DaysBack'] = (config['Stop'] - config['Start']).days
    config['HPSamplingTime'] = env_params['HPSamplingTime']
    config['Hour_MPC_Horizon'] = env_params['Hour_MPC_Horizon']  # MPC length in hours
    config['N_MPC_Horizon'] = int(60 * config['Hour_MPC_Horizon'] / config['HPSamplingTime'])
    ### leave empty if you do not have m27 installed!
    ### m27 as opposed to cut runtime
    # config['solver_options'] = {'linear_solver': 'ma27',
    #                             'hsllib': '/Users/kqiu/ThirdParty-HSL/.libs/libcoinhsl.dylib'}
    #

    ### get timesteps in datetime
    timestep = config['Start']
    config['timesteps'] = [timestep]
    config['timesteps+N'] = [timestep]
    while timestep < (config['Stop'] - dt.timedelta(minutes=5)):
        timestep += dt.timedelta(minutes=5)
        config['timesteps'].append(timestep)
        config['timesteps+N'].append(timestep)
    while timestep <= (config['Stop'] + dt.timedelta(hours=config['Hour_MPC_Horizon']) - dt.timedelta(minutes=5)):
        timestep += dt.timedelta(minutes=5)
        config['timesteps+N'].append(timestep)

    ### get timesteps in index
    config['dt'] = []
    config['dt+N'] = []
    for ts in config['timesteps']:
        delta = ts - config['timesteps'][0]
        config['dt'].append(delta.total_seconds() / 60)
    for ts in config['timesteps+N']:
        delta = ts - config['timesteps+N'][0]
        config['dt+N'].append(delta.total_seconds() / 60)

    ### Simulation noise
    config['mu'] = env_params['mu']
    config['sigma'] = env_params['sigma']

    ### Get RL cost weights
    config['cost_weights'] = {}
    for key in env_params['cost_weights'].keys():
        config['cost_weights'][key] = env_params['cost_weights'][key]

    return config


def prediction_functions(ReLu, COP0, COP1, T0):
    ### define power saturation function
    pow = MX.sym("pow")
    maxpow = MX.sym("maxpow")
    Pow = log(1 + exp(ReLu * pow)) / ReLu
    Pow = Pow - log(1 + exp(ReLu * (Pow - maxpow))) / ReLu
    PowSat_func = Function('PowSat', [pow, maxpow], [Pow])

    ### define COP function
    OutTemp = MX.sym('OutTemp')
    COP = COP0 - np.log(1 + np.exp(ReLu * COP1 * (T0 - OutTemp))) / ReLu
    COP_func = Function('COP', [OutTemp], [COP])

    ### define prediction model
    Wall = MX.sym('Wall')
    Room = MX.sym('Room')
    TargetTemp = MX.sym('TargetTemp')
    Fan = MX.sym('Fan')
    OutTemp = MX.sym('OutTemp')
    HPset = MX.sym('HPset')
    HPin = MX.sym('HPin')
    HP0 = MX.sym('HP0')
    FanParam = MX.sym('FanParam')
    m_wall = MX.sym('m_wall')
    m_air = MX.sym('m_air')
    rho_out = MX.sym('rho_out')
    rho_in = MX.sym('rho_in')
    rho_dir = MX.sym('rho_dir')

    ### Power prediction
    DT = HPset * TargetTemp
    DT -= HPin * Room
    Pow = exp(FanParam * (Fan - 5)) * (DT + HP0)
    Pow_func = Function('Pow', [HPset, HPin, HP0, FanParam, TargetTemp, Room, Fan], [Pow])

    ### Wall Temperature
    WallPlus = m_wall * Wall + rho_out * (OutTemp - Wall) + rho_in * (Room - Wall)
    WallPlus_func = Function('WallPlus', [m_wall, rho_out, rho_in, Wall, Room, OutTemp], [WallPlus])

    ### Room Temperature
    COP = MX.sym('COP')
    Pow = MX.sym('Pow')
    RoomPlus = m_air * Room + rho_in * (Wall - Room) + rho_dir * (OutTemp - Room) + COP * Pow
    RoomPlus_func = Function('RoomPlus', [m_air, rho_in, rho_dir, Room, Wall, OutTemp, COP, Pow], [RoomPlus])

    return PowSat_func, COP_func, Pow_func, WallPlus_func, RoomPlus_func


class House4Pumps:
    def __init__(self, env_params: dict) -> None:
        ### process data file
        print('---- building parameter [self.config] and input data [self.InputDict]')
        start = datetime.now()
        self.config = GetConfig(env_params)
        HPutils2 = HPutils(self.config['SensiboDevices'], self.config['DataPath'])
        DR = DataReader(self.config['DataPath'], HPutils2, self.config)
        self.InputDict = DR.get_input_dictionary(self.config)
        print(f"---- done building works in {(datetime.now() - start).seconds} seconds-----")

        ###  some settings
        self.pumps = self.config['SensiboDevices']
        self.HPSamplingTime = self.config['HPSamplingTime']
        self.Hour_MPC_Horizon = self.config['Hour_MPC_Horizon']
        self.params = self.InputDict['params']
        self.scale = self.InputDict['scale']
        self.COP0, self.COP1, self.T0 = self.InputDict['COP']
        self.mu = self.config['mu']
        self.sigma = self.config['sigma']
        self.GridCost = 44
        self.ReLu = 100
        self.PowerHistogram = self.InputDict['PowerHistogram']
        self.TempHistogram = self.InputDict['TempHistogram']
        self.MaxPow = {'main': 1.,
                       'living': 1.5,
                       'livingdown': 1.,
                       'studio': 1.}
        self.cost_weights = self.config['cost_weights']
        self.PowSat, self.COP, self.Pow, self.WallPlus, self.RoomPlus = prediction_functions(
            self.ReLu, self.COP0, self.COP1, self.T0)

        ### define state box and action box in dict format
        self.observation_space = {}
        self.action_space = {}
        for pump in self.pumps:
            self.observation_space[pump] = {"Room": {'low': 0, 'high': 35},
                                            "Wall": {'low': 0, 'high': 35},
                                            "TargetTemp": {'low': 0, 'high': 35},
                                            "Fan": {'low': 0, 'high': 5}
                                            }

            self.action_space[pump] = {"Delta_TargetTemp": {'low': -35, 'high': 35},
                                       "Delta_Fan": {'low': -5, 'high': 5},
                                       # "On": {'low': 0, 'high': 1}
                                       }

        ### generate all the needed data in dict format
        self.data = {
            'OutTemp': self.InputDict['OutTemp'],
            'Spot': self.InputDict['Spot'],
            'DesiredTemp': {},
            'MinTemp': {},
            'MaxFan': {}
        }
        for pump in self.pumps:
            self.data['DesiredTemp'][pump] = self.InputDict['DesiredTemp'][pump]
            self.data['MinTemp'][pump] = self.InputDict['MinTemp'][pump]
            self.data['MaxFan'][pump] = self.InputDict['MaxFan'][pump]

        ### design state and action structure
        state = struct_symMX([entry('Room'),
                              entry('Wall'),
                              entry('TargetTemp'),
                              entry('Fan')])
        action = struct_symMX([entry('Delta_TargetTemp'),
                               entry('Delta_Fan'),
                               # entry('On')
                               ])
        state_all = []
        action_all = []
        for pump in self.pumps:
            state_all += [entry(pump, struct=state)]
            action_all += [entry(pump, struct=action)]
        self.state_struct = struct_symMX(state_all)
        self.action_struct = struct_symMX(action_all)

        ### time stamp
        self.t = 0

        ### initialization
        self.state = self.state_struct(0)
        self.reset()

    def reset(self):
        for pump in self.pumps:
            self.state[pump, 'Room'] = self.InputDict['state0'][pump]["Room"] + 0 * 0.5 * np.random.normal(),
            self.state[pump, 'Wall'] = self.InputDict['state0'][pump]["Wall"] + 0 * 0.5 * np.random.normal(),
            self.state[pump, 'TargetTemp'] = self.InputDict['state0'][pump]["TargetTemp"] + 0 * 0.5 * np.random.normal(),
            self.state[pump, 'Fan'] = self.InputDict['state0'][pump]["Fan"] + 0 * 0.5 * np.random.normal()

            ### clip
            self.state[pump, 'Room'] = np.clip(self.state[pump, 'Room'],
                                               self.observation_space[pump]['Room']['low'],
                                               self.observation_space[pump]['Room']['high'])
            self.state[pump, 'Wall'] = np.clip(self.state[pump, 'Wall'],
                                               self.observation_space[pump]['Wall']['low'],
                                               self.observation_space[pump]['Wall']['high'])
            self.state[pump, 'TargetTemp'] = np.clip(self.state[pump, 'TargetTemp'],
                                                     self.observation_space[pump]['TargetTemp']['low'],
                                                     self.observation_space[pump]['TargetTemp']['high'])
            self.state[pump, 'Fan'] = np.clip(self.state[pump, 'Fan'],
                                              self.observation_space[pump]['Fan']['low'],
                                              self.observation_space[pump]['Fan']['high'])

        ### reset time
        self.t = 0

        return self.state

    ### real model used in env.step
    def model_real(self, state: DM, action: DM, data: dict):
        next_state = self.state_struct(0)
        COP = self.COP(data['OutTemp'])

        PowTot = 0
        Power = {}
        for pump in self.pumps:
            Pow = self.Pow(self.params['HPset'][pump],
                           self.params['HPin'][pump],
                           self.params['HP0'][pump],
                           self.params['Fan'][pump],
                           state[pump, 'TargetTemp'] + action[pump, 'Delta_TargetTemp'],
                           state[pump, 'Room'],
                           state[pump, 'Fan'] + action[pump, 'Delta_Fan'])
            # Pow *= action[pump, 'On']
            Power[pump] = Pow
            PowTot += Pow

            WallPlus = self.WallPlus(self.params['m_wall'][pump],
                                     self.params['rho_out'][pump],
                                     self.params['rho_in'][pump],
                                     state[pump, 'Wall'],
                                     state[pump, 'Room'],
                                     data['OutTemp'])

            RoomPlus = self.RoomPlus(self.params['m_air'][pump],
                                     self.params['rho_in'][pump],
                                     self.params['rho_dir'][pump],
                                     state[pump, 'Room'],
                                     state[pump, 'Wall'],
                                     data['OutTemp'],
                                     COP,
                                     Pow)

            RoomPlus *= 1 / self.params['m_air'][pump]
            WallPlus *= 1 / self.params['m_wall'][pump]

            ### additive noise
            RoomPlus += np.random.choice(
                self.TempHistogram[pump]['centers'], p=self.TempHistogram[pump]['probability'])

            next_state[pump, 'Room'] = RoomPlus.full().squeeze()
            next_state[pump, 'Wall'] = WallPlus.full().squeeze(),
            next_state[pump, 'TargetTemp'] = state[pump, 'TargetTemp'] + action[pump, 'Delta_TargetTemp']
            next_state[pump, 'Fan'] = state[pump, 'Fan'] + action[pump, 'Delta_Fan']

        ### additive noise ??? The noised version is useless
        # PowNoise = {}
        # PowNoise['relative'] = np.random.choice(
        #     self.PowerHistogram['centers'], p=self.PowerHistogram['probability'])
        # PowNoise['absolute'] = PowNoise['relative'] * PowTot
        # for pump in self.pumps:
        #     scale = Power[pump] / PowTot
        #     PowNoise[pump] = scale * PowNoise['absolute']
        #     Power[pump] = Power[pump] + PowNoise[pump]

        power_info = {
            'power_total': PowTot,
            'power_pump': Power
        }

        return next_state, power_info

    def reward_fn(self, state: DM, action: DM, data: dict, PowTot: float):
        l_temp = 0

        for pump in self.pumps:
            l_temp += self.cost_weights['TempAbove'] * (data['DesiredTemp'][pump] - state[pump, 'Room']) ** 2

            if state[pump, 'Room'] < data['DesiredTemp'][pump]:
                l_temp += self.cost_weights['TempBelow'] * (data['DesiredTemp'][pump] - state[pump, 'Room']) ** 2

            if state[pump, 'Room'] < data['MinTemp'][pump]:
                l_temp += self.cost_weights['MinTemp'] * (data['MinTemp'][pump] - state[pump, 'Room']) ** 2

            l_temp += self.cost_weights['Delta_TargetTemp'] * action[pump, 'Delta_TargetTemp'] ** 2

            l_temp += self.cost_weights['Delta_Fan'] * action[pump, 'Delta_Fan'] ** 2

        l_spot = self.cost_weights['SpotGain'] * (data['Spot'] + self.GridCost) * PowTot

        reward = (l_spot + l_temp).full().squeeze()
        done = 0

        return reward, done

    def cost_fn(self):
        pass

    def step(self, action: DM):
        ### generate current data at time index self.t
        data = {
            'OutTemp': self.data['OutTemp'][self.t],
            'Spot': self.data['Spot'][self.t],
            'DesiredTemp': {},
            'MinTemp': {},
            'MaxFan': {}
        }
        for pump in self.pumps:
            data['DesiredTemp'][pump] = self.data['DesiredTemp'][pump][self.t]
            data['MinTemp'][pump] = self.data['MinTemp'][pump][self.t]
            data['MaxFan'][pump] = self.data['MaxFan'][pump][self.t]

        next_state, power_info = self.model_real(self.state, action, data)

        for pump in self.pumps:
            next_state[pump, 'Room'] = np.clip(next_state[pump, 'Room'],
                                               self.observation_space[pump]['Room']['low'],
                                               self.observation_space[pump]['Room']['high'])
            next_state[pump, 'Wall'] = np.clip(next_state[pump, 'Wall'],
                                               self.observation_space[pump]['Wall']['low'],
                                               self.observation_space[pump]['Wall']['high'])
            next_state[pump, 'TargetTemp'] = np.clip(next_state[pump, 'TargetTemp'],
                                                     self.observation_space[pump]['TargetTemp']['low'],
                                                     self.observation_space[pump]['TargetTemp']['high'])
            next_state[pump, 'Fan'] = np.clip(next_state[pump, 'Fan'],
                                              self.observation_space[pump]['Fan']['low'],
                                              self.observation_space[pump]['Fan']['high'])

        reward, done = self.reward_fn(self.state, action, data, power_info['power_total'])

        # print(f'reward = {reward}')
        # print(f'PowTot = {power_info['power_total']}')

        self.state = next_state
        ### time index increase
        self.t += 1

        return self.state, reward, done, power_info


### test
if __name__ == "__main__":
    json_path = os.path.abspath(os.path.join(os.getcwd(), '../../Settings/other/house_4pumps_rl_mpc_lstd.json'))
    with open(json_path, 'r') as f:
        params = json.load(f)
        params["env_params"]["json_path"] = json_path
        print(f'env_params = {params}')
    a = House4Pumps(params["env_params"])
    print(a.state)
    print(f'initial state = {a.state.cat}')
    print(f'initial time = {a.t}')
    print("------ one step forward ------")
    action = a.action_struct([1, 1, 0.1, 1, 1, 0.1, 1, 1, 0.1, 1, 1, 0.1])
    next_state, reward, done, info = a.step(action)
    print(f'current state = {next_state.cat}')
    print(f'reward = {reward}')
    print(f'done = {done}')
    print(f'info = {info}')
    print(f'current time = {a.t}')
