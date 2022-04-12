import numpy as np
from datetime import date, datetime, timedelta, tzinfo, time
import pickle
from casadi import *
from casadi.tools import *
import matplotlib.pyplot as plt
import pandas as pd


class Plotter:
    def __init__(self) -> None:
        f = open('Results/House4Pumps/Results.pkl', "rb")
        self.results = pickle.load(f)
        f.close()

        self.input = self.results['input']
        self.config = self.results['config']
        self.buffer = self.results['buffer']

        self.timesteps = self.results['config']['timesteps']
        self.pumps = self.input['pumps']
        self.steps = len(self.timesteps)

        state_batch, action_batch, reward_batch, next_state_batch, done_batch, info_batch = \
            self.buffer.sample_sequence(self.steps)

        self.states = {}
        self.powers = {}
        for pump in self.pumps:
            self.states[pump] = {'Room': [],
                                 'Wall': [],
                                 'TargetTemp': [],
                                 'Fan': []}
            self.powers[pump] = []

        num_pumps = len(self.pumps)
        for i in range(self.steps):
            for j, pump in enumerate(self.pumps):
                self.states[pump]['Room'].append(state_batch[i][j*num_pumps].full().squeeze())
                self.states[pump]['Wall'].append(state_batch[i][j*num_pumps+1].full().squeeze())
                self.states[pump]['TargetTemp'].append(state_batch[i][j*num_pumps+2].full().squeeze())
                self.states[pump]['Fan'].append(state_batch[i][j*num_pumps+3].full().squeeze())

                self.powers[pump].append(info_batch[i]['power_info']['power_pump'][pump].full().squeeze())

        self.desired_temp = {}
        for pump in self.pumps:
            self.desired_temp[pump] = self.input['DesiredTemp'][pump][:self.steps]

        self.spot = self.input['Spot'][:self.steps] * np.array(0.1)


    def plot(self):
        fig1, ax1 = plt.subplots(2, 2)
        # plot main
        ax1[0, 0].plot(self.timesteps[:], self.desired_temp['main'][:], label='desired temperature')
        ax1[0, 0].plot(self.timesteps[:], self.states['main']['Room'][:], label='room temperature')
        ax1[0, 0].plot(self.timesteps[:], self.states['main']['TargetTemp'][:], label='target temperature')
        ax1[0, 0].plot(self.timesteps[:], self.spot[:], label='spot price')
        ax1[0, 0].title.set_text('main')
        ax1[0, 0].tick_params(labelrotation=45)

        # living
        ax1[0, 1].plot(self.timesteps[:], self.desired_temp['living'][:], label='desired temperature')
        ax1[0, 1].plot(self.timesteps[:], self.states['living']['Room'][:], label='room temperature')
        ax1[0, 1].plot(self.timesteps[:], self.states['living']['TargetTemp'][:], label='target temperature')
        ax1[0, 1].plot(self.timesteps[:], self.spot[:], label='spot price')
        ax1[0, 1].title.set_text('living')
        ax1[0, 1].tick_params(labelrotation=45)

        # studio
        ax1[1, 0].plot(self.timesteps[:], self.desired_temp['studio'][:], label='desired temperature')
        ax1[1, 0].plot(self.timesteps[:], self.states['studio']['Room'][:], label='room temperature')
        ax1[1, 0].plot(self.timesteps[:], self.states['studio']['TargetTemp'][:], label='target temperature')
        ax1[1, 0].plot(self.timesteps[:], self.spot[:], label='spot price')
        ax1[1, 0].title.set_text('studio')
        ax1[1, 0].tick_params(labelrotation=45)

        # livingdown
        ax1[1, 1].plot(self.timesteps[:], self.desired_temp['livingdown'][:], label='desired temperature')
        ax1[1, 1].plot(self.timesteps[:], self.states['livingdown']['Room'][:], label='room temperature')
        ax1[1, 1].plot(self.timesteps[:], self.states['livingdown']['TargetTemp'][:], label='target temperature')
        ax1[1, 1].plot(self.timesteps[:], self.spot[:], label='spot price')
        ax1[1, 1].title.set_text('livingdown')
        ax1[1, 1].tick_params(labelrotation=45)

        # one legend for all sugplots
        handles, labels = ax1[0, 0].get_legend_handles_labels()
        fig1.legend(handles, labels, loc='upper center')

        ##########################
        ##########################
        fig2, ax2 = plt.subplots(2, 2)
        # plot main
        ax2[0, 0].plot(self.timesteps[:], self.states['main']['Fan'][:], label='Fan')
        ax2[0, 0].plot(self.timesteps[:], self.powers['main'][:], label='Power')
        ax2[0, 0].title.set_text('main')
        ax2[0, 0].tick_params(labelrotation=45)

        # living
        ax2[0, 1].plot(self.timesteps[:], self.states['living']['Fan'][:], label='Fan')
        ax2[0, 1].plot(self.timesteps[:], self.powers['living'][:], label='Power')
        ax2[0, 1].title.set_text('living')
        ax2[0, 1].tick_params(labelrotation=45)

        # studio
        ax2[1, 0].plot(self.timesteps[:], self.states['studio']['Fan'][:], label='Fan')
        ax2[1, 0].plot(self.timesteps[:], self.powers['studio'][:], label='Power')
        ax2[1, 0].title.set_text('studio')
        ax2[1, 0].tick_params(labelrotation=45)

        # livingdown
        ax2[1, 1].plot(self.timesteps[:], self.states['livingdown']['Fan'][:], label='Fan')
        ax2[1, 1].plot(self.timesteps[:], self.powers['livingdown'][:], label='Power')
        ax2[1, 1].title.set_text('livingdown')
        ax2[1, 1].tick_params(labelrotation=45)

        # one legend for all sugplots
        handles, labels = ax2[0, 0].get_legend_handles_labels()
        fig2.legend(handles, labels, loc='upper center')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    Plotter = Plotter()
    Plotter.plot()

