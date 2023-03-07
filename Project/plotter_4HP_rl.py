import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Plotter:
    def __init__(self, open_file_name, save_fig_name):
        self.open_file_name = open_file_name
        self.save_fig_name = save_fig_name

        f = open(self.open_file_name, "rb")
        results = pickle.load(f)
        f.close()

        self.mpc_thetas = results['theta_log']
        self.eval_returns = results['eval_returns']

        self.iteration = len(self.mpc_thetas)
        self.num_theta = self.mpc_thetas[0].size

        self.theta = np.zeros((self.num_theta, self.iteration))
        for ite in range(self.iteration):
            for index_theta in range(self.num_theta):
                self.theta[index_theta, ite] = self.mpc_thetas[ite][index_theta]

        self.theta_step = list(range(self.iteration))
        self.eval_step = list(range(len(self.eval_returns)))

    def plot(self):
        fig1, ax1 = plt.subplots(3, 4)
        for i in range(11):
            ax1[i // 4, i % 4].plot(self.theta_step, self.theta[i, :])
        ax1[2, 3].plot(self.eval_step, self.eval_returns)
        # one legend for all sugplots
        # handles, labels = ax1[0, 0].get_legend_handles_labels()
        # fig1.legend(handles, labels, loc='upper center')

        plt.title(self.save_fig_name)
        plt.tight_layout()
        # plt.show()
        plt.savefig(self.save_fig_name)


if __name__ == '__main__':
    Plotter = Plotter('Results/House4Pumps/Results_rl.pkl', 'Results/House4Pumps/Results_rl_fig.png')
    Plotter.plot()
