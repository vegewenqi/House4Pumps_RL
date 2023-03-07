# standard libraries
import sys
import math
import pickle as pkl
from random import randint

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as Functional
import torch.optim as optim
from matplotlib.animation import FuncAnimation

"""
TODO for cleaning up code:
- Remove all not useful comments
- double check that we use class variables correctly and all places they can be used
- set weights for loss functions
    10.0 seems to work good for decay
"""

dev = "cpu"
# Check if we can run on GPU
if torch.cuda.is_available():
    dev = "cuda:0"
device = torch.device(dev)


class DMPC():
    def __init__(self, p=0, m=0, N=1, H=1, filename="", overwrite=False, hist_batch=[], u_batch=[], y_batch=[],
                 lr=0.001, D_reg_weight=1000.0, time_reg_weight=100.0, allIO={}, convIO={}):
        self.p = p  # dimension of outputs
        self.m = m  # dimension of inputs
        self.N = N  # horizon length
        self.H = H  # history length

        self.best_loss = float('inf')
        self.hist_batch = hist_batch
        self.u_batch = u_batch
        self.y_batch = y_batch
        self.filename = filename
        self.overwrite = overwrite

        self.u, self.y = [], []
        self.block_indeces = []
        self.losses = []
        self.losses_show_last = 20

        self.lr = lr
        self.D_reg_weight = D_reg_weight
        self.time_reg_weight = time_reg_weight

        # dictionaries for overview of inputs and outputs
        self.allIO = allIO  # dictionaries with keys 'Act' and 'Obs' with values of lists of strings
        self.convIO = convIO

        self.updatePhi()
        # ? Should seed for torch and numpy be added to class?

    # * update class variables
    def updatePhi(self):
        '''construct tensor of size equal to elements of non-zero elements in Phi'''
        self.phi_c = 0.1 * torch.rand(
            int(self.p * (self.p + self.m) * self.N * self.H + self.p * self.m * self.N * (self.N + 1) / 2))
        self.phi_c.requires_grad_()
        self.phi_c.to(device)

    def add_u(self, u):
        '''Add a input/control/action data (state-space u)'''
        dataCheck(u)
        self.u.append(np.array(u, dtype=object))
        self.m += 1
        self.updatePhi()

    def add_y(self, y):
        '''Add a measurement/output data (state-space y)'''
        dataCheck(y)
        self.y.append(np.array(y, dtype=object))
        self.p += 1
        self.updatePhi()

    def set_phi_c(self, phi_c):
        self.phi_c = phi_c

    # * formula functions
    def phi_fun(self, P, lib=torch):
        '''Building phi as torch 2d array'''
        P_dim = (self.N * self.p, (self.m + self.p) * self.H)
        F_dim = (self.N * self.p, self.m * self.N)

        index_P = 0
        Phi_P = lib.zeros(P_dim)
        for col in range(P_dim[-1]):
            for row in range(P_dim[0]):
                Phi_P[row, col] = P[index_P]
                index_P += 1

        Phi_F = lib.zeros(F_dim)
        c = 0
        d_row = -1
        u, y = [0] * self.p * self.N, [0] * self.m * self.N  # row, and col counter list
        for col in range(F_dim[-1]):
            for row in range(F_dim[0]):
                if row > d_row:
                    Phi_F[row, col] = P[P_dim[0] * P_dim[-1] + c]
                    c += 1
                    if u[row] < self.m and y[col] < self.p:
                        u[row] += 1
                        y[col] += 1
                    if u[row] == self.m and y[col] == self.p:
                        d_row = row

        # check if lib is torch or numpy
        if lib == torch:
            Phi = torch.cat((Phi_P, Phi_F), 1)
        elif lib == np:
            Phi = np.concatenate((Phi_P, Phi_F), 1)
        else:
            raise Exception("lib must be torch or numpy (np)")

        return Phi

    # * Regularization functions
    def find_block_indeces(self):
        past = (self.p + self.m) * self.H * self.p * self.N
        i = 0
        u = [0] * self.p * self.N
        y = [0] * self.m * self.N
        i_row = -1  # index row limit
        D_blocks, temp = [], []  # List of lists of indexes
        c = 0
        bellow_D_indexes = []
        for col in range(self.m * self.N):
            for row in range(self.p * self.N):
                if row > i_row:
                    i += 1
                    # First D_matrix
                    if u[row] < self.m and y[col] < self.p:
                        u[row] += 1
                        y[col] += 1
                        temp.append(past + i - 1)
                        c += 1
                        if c == self.m * self.p:
                            D_blocks.append(temp)
                            temp, c = [], 0
                    else:
                        bellow_D_indexes.append(past + i - 1)
                    # Keeping track if index limit
                    if u[row] == self.m and y[col] == self.p:
                        i_row = row
        blocks = []
        temp = []
        i = self.N - 1
        for _ in range(self.N - 1):  # Diagonal boxes
            temp = []
            for _ in range(i): temp.append([])  # boxes on each diagonal
            blocks.append(temp)
            i -= 1

        y, u, col, diag, blockNum = 0, 0, 0, 0, 0
        for index in bellow_D_indexes:
            blocks[diag][blockNum].append(index)
            y += 1
            if y >= self.p:
                y = 0
                diag += 1
                if diag + col > self.N - 2:
                    u += 1
                    if u < self.m:
                        diag = 0
                    else:
                        u, diag = 0, 0
                        blockNum += 1
                        col += 1

        indexes = []
        indexes.append(D_blocks)
        for diag in blocks:
            indexes.append(diag)
        self.block_indeces = indexes
        return indexes

    def time_impact(self):
        '''
        Set the impact of the different inputs and outputs on the prediction.
        The impact decays from right to left.
        This impact factor is only set to the past inputs and outputs, not the future inputs.

        Returns a matrix of the same size as Phi_P, where the impact of each past input and output is scaled.
        '''

        """
        #! The Phi have to be sorted by time index, not output by output and input by input.
        E.g, two inputs (u and v), two outputs (x and z), and H=3:
        u = [u_-2, u_-1, u_0]
        v = [v_-2, v_-1, v_0]
        x = [x_-2, x_-1, x_0]
        z = [z_-2, z_-1, z_0]

        Then coloumns of Phi must be structured as follow for each row in Phi_P (impact factor is an example):
        y_k+1           =[u_-2    v_-2    u_-1     v_-1    u_0     v_0|   x_-2    z_-2     x_-1    z_-1    x_0     z_0 || Phi_f ->]

        impact fator    =[1        1     0.5      0.5     0.25    0.25|   1       1       0.5     0.5     0.25    0.25 || 1 ->]
        """

        """
        !The factor is higher from left to right, even though importance is from right to left.
        !This is because when minimizing and using L1, we want to reduce terms to the left more than terms to the right. 
        """

        time_impact = torch.ones((self.p * self.N, (self.m + self.p) * self.H))
        decayFactor = 1 / (self.H / 2)

        for row in range(self.p * self.N):
            for timeStamp in range(self.H):

                for input in range(self.m):
                    # exponential decay
                    time_impact[row][timeStamp * self.m + input] = math.exp(-timeStamp * decayFactor)

                for output in range(self.p):
                    time_impact[row][self.m * self.H + timeStamp * self.p + output] = math.exp(-timeStamp * decayFactor)

        return time_impact

    def convection_impact(self):
        '''
        Set the impact of the different inputs and outputs on eachother based on convection.
        The impact decays from right to left.

        Returns a truth matrix of the same size as Phi.
        '''

        """
        E.g, two inputs (u and v), two outputs (x and z) and u is input to x and v is input to z, and H=3, N=2:
        u = [u_-2, u_-1, u_0]
        v = [v_-2, v_-1, v_0]
        x = [x_-2, x_-1, x_0]
        z = [z_-2, z_-1, z_0]

        Then coloumns of Phi must be structured as follow for each row in Phi_P (impact factor is an example):
        x_1 = | u_-2    v_-2    u_-1     v_-1    u_0     v_0|   x_-2    z_-2     x_-1    z_-1    x_0     z_0 || u_1    v_1    u_2    v_2 | 
        z_1 = | u_-2    v_-2    u_-1     v_-1    u_0     v_0|   x_-2    z_-2     x_-1    z_-1    x_0     z_0 || u_1    v_1    u_2    v_2 |

        Then the convection impact factor is:
        x_1 = | 0       1       0       1       0       1|   0       1       0       1       0       1 || 0     1     0     1 |
        z_1 = | 1       0       1       0       1       0|   1       0       1       0       1       0 || 1     0     1     0 |
        x_2 = | 0       1       0       1       0       1|   0       1       0       1       0       1 || 0     1     0     1 |
        z_2 = | 1       0       1       0       1       0|   1       0       1       0       1       0 || 1     0     1     0 |

        !This structure wil penalize impact from v and z on x, and u and x on z, through L1 minimization at a later step.
        """

        P_dim = (self.N * self.p, (self.m + self.p) * self.H)
        F_dim = (self.N * self.p, self.m * self.N)

        # build phi structure using allIO and convIO, to create truth matrix
        allObs = self.allIO['Obs']
        allAct = self.allIO['Act']
        convObs = self.convIO['Obs']
        convAct = self.convIO['Act']

        conv_impact = torch.zeros(
            int(self.p * (self.p + self.m) * self.N * self.H + self.p * self.m * self.N * (self.N + 1) / 2))
        conv_impact = self.phi_fun(conv_impact)

        # create string version of Phi matrix
        phi_structure_P = np.empty((self.p * self.N, self.H * (self.m + self.p)), dtype=object)

        # fill in phi_structure_P matrix with all outputs and inputs
        for row in range(P_dim[0]):
            for timeStamp in range(self.H):
                for input in range(self.m):
                    phi_structure_P[row][timeStamp * self.m + input] = allAct[input]
                for output in range(self.p):
                    phi_structure_P[row][self.m * self.H + timeStamp * self.p + output] = allObs[output]

        action = 0
        d_row = -1
        phi_structure_F = np.empty((self.p * self.N, self.N * self.m), dtype=object)

        u, y = [0] * self.p * self.N, [0] * self.m * self.N  # row, and col counter list
        for col in range(F_dim[-1]):
            for row in range(F_dim[0]):
                if row > d_row:
                    phi_structure_F[row, col] = allAct[action]
                    if u[row] < self.m and y[col] < self.p:
                        u[row] += 1
                        y[col] += 1
                    if u[row] == self.m and y[col] == self.p:
                        d_row = row

            action += 1
            if (col + 1) % self.m == 0:  # check if action counter should be reset on the next coloumn
                action = 0

        convection_impact = np.concatenate((phi_structure_P, phi_structure_F), axis=1)
        np.savetxt('Phi_structure.txt', convection_impact, fmt='%s', delimiter=' \t \t')

        # *Create truth matrix
        nRows = self.p * self.N
        nPastInputs = self.m * self.H
        nPastOutputs = self.p * self.H
        nPast = nPastInputs + nPastOutputs
        nFutInputs = self.m * self.N

        for row in range(0, nRows, self.p):
            yInd = 0
            for y in allObs:

                if y in convObs:  # if the observation is affected by convection
                    y_prefix = y.split('_')[0]

                    # Phi_P part
                    for pastAct in range(0, nPastInputs, self.m):
                        pastActInd = 0
                        for act in allAct:

                            if act in convAct:  # if the action affects convection
                                act_prefix = act.split('_')[0]

                                if y_prefix == act_prefix:
                                    convection_impact[row + yInd][pastAct + pastActInd] = 0
                                else:
                                    convection_impact[row + yInd][
                                        pastAct + pastActInd] = 1  # set it to 1 as we want to penalize the impact from other inputs on this output

                            else:  # action does not affect convection
                                convection_impact[row + yInd][pastAct + pastActInd] = 0

                            pastActInd += 1

                    for pastObs in range(nPastInputs, nPast, self.p):
                        pastObsInd = 0
                        for obs in allObs:
                            if obs in convObs:  # if the observation is affected by convection
                                y_prefix = y.split('_')[0]
                                obs_prefix = obs.split('_')[0]

                                if y_prefix == obs_prefix:
                                    convection_impact[row + yInd][pastObs + pastObsInd] = 0
                                else:
                                    convection_impact[row + yInd][pastObs + pastObsInd] = 1

                            else:  # observation does not affect convection
                                convection_impact[row + yInd][pastObs + pastObsInd] = 0

                            pastObsInd += 1

                    # Phi_F part
                    for futureAct in range(nPast, nPast + nFutInputs, self.m):
                        if convection_impact[row + yInd][futureAct] == None:  # end of block
                            convection_impact[row + yInd][futureAct:] = 0
                            break

                        futureActInd = 0
                        for act in allAct:
                            if act in convAct:
                                act_prefix = act.split('_')[0]

                                if y_prefix == act_prefix:
                                    convection_impact[row + yInd][futureAct + futureActInd] = 0

                                else:
                                    convection_impact[row + yInd][futureAct + futureActInd] = 1

                            else:  # action does not affect convection
                                convection_impact[row + yInd][futureAct + futureActInd] = 0

                            futureActInd += 1


                else:  # row is not affected by convection, so should not be penalized
                    convection_impact[row + yInd] = 0

                yInd += 1

        np.savetxt('Phi_truth.txt', convection_impact, fmt='%s', delimiter='\t \t')
        convection_impact = torch.from_numpy(convection_impact.astype(np.int32))

        return convection_impact

    def block_regularization(self, phi, error_exponent: int = 2, error_weight=1.0):
        '''
        phi: phi as 1d array

        Returns the chained sum of the deviation between items in the block-matrixes along the diagonal of phi_F multiplied by error_weight
        '''
        sum = 0
        # for every diagonal
        for d in range(len(self.block_indeces)):
            if len(self.block_indeces[d]) > 1:
                # For every item in every block
                for item in range(len(self.block_indeces[d][0])):
                    for mat in range(len(self.block_indeces[d]) - 1):
                        # Get indexes
                        current = self.block_indeces[d][mat][item]
                        next = self.block_indeces[d][mat + 1][item]
                        # Error
                        sum += error_weight * (phi[current] - phi[next]) ** error_exponent
        return sum

    def decaying_impact_reguralization(self, phi, time_impact, error_weight=10.0):
        '''
        for each row in phi, let the impact of each column decay with time
        the latest column has the highest impact, the first column has the lowest impact

         phi: phi as 1d array

         returns L1 norm of impact-regulated Phi_P matrix
        '''

        P_dim = (self.p * self.N, (self.m + self.p) * self.H)
        index = 0
        Phi_P = torch.zeros(P_dim)
        for col in range(P_dim[-1]):
            for row in range(P_dim[0]):
                Phi_P[row, col] = phi[index]
                index += 1

        decay_penalty = time_impact * Phi_P

        # L1 norm of each coloumn in decay_penalty
        L1_coloumns = torch.linalg.norm(decay_penalty, ord=1, dim=0)
        sumL1 = L1_coloumns.sum()

        return error_weight * sumL1

    def convection_impact_reguralization(self, phi, convection_impact, error_weight=10.0):

        Phi = self.phi_fun(phi)

        convection_penalty = convection_impact * Phi

        L1_coloumns = torch.linalg.norm(convection_penalty, ord=1, dim=0)
        sumL1 = L1_coloumns.sum()

        return error_weight * sumL1

    # * Optimize Phi
    def build_batches(self):
        # Prepare data
        self.n_data = self.u[0].shape[0]
        self.n_batches = self.n_data - self.N - self.H

        u = np.zeros((self.m, self.n_data))
        y = np.zeros((self.p, self.n_data))

        rows, cols = u.shape
        for r in range(rows):
            row = self.u[r]
            for c in range(cols):
                u[r, c] = row[c]

        rows, cols = y.shape
        for r in range(rows):
            row = self.y[r]
            for c in range(cols):
                y[r, c] = row[c]

        self.hist_batch = []
        self.u_batch = []
        self.y_batch = []

        # Build batches
        print("Building data, " + str(self.n_data - self.N - self.H) + " batches")

        for i in range(self.n_data - self.N - self.H):
            # History
            arr = np.concatenate((
                np.array(u[:, i:i + self.H]).T.reshape(-1),
                np.array(y[:, i:i + self.H]).T.reshape(-1)
            ))
            self.hist_batch.append(arr)
            # Future
            self.u_batch.append(np.array(u[:, i + self.H: i + self.H + self.N]).T.reshape(-1))
            self.y_batch.append(np.array(y[:, i + self.H: i + self.H + self.N]).T.reshape(-1))

    def calc_phi(self, train_it=20, D_reg_weight=100.0, time_reg_weight=10.0, conv_reg_weight=10.0, lr=0.001,
                 live_plt=False, block_reg=True, decay_reg=True, conv_reg=False):
        '''Gradient descent to find optimized phi'''

        self.phi_opt = optim.Adam([self.phi_c], lr=lr)

        # Load from file if a path is given and we are not to overwrite
        if not self.filename == "" and not self.overwrite:
            print("Loading from path:", self.filename)
            self.load_dmpc(self.filename)
            f = self.filename.split(".")
            sname = f[0].split("_")
            if len(sname) > 1:
                if sname[1] == "best":
                    self.filename = sname[0] + "." + f[1]
                    print(" loading from 'best' file, will save in " + self.filename)

        if live_plt:
            self.live_plot_init()

        if block_reg:
            self.find_block_indeces()

        if decay_reg:
            time_impact = self.time_impact()

        if conv_reg:
            convection_impact = self.convection_impact()

        self.n_batches = len(self.hist_batch)
        print("n_batches:", self.n_batches)
        if not self.hist_batch or not self.u_batch or not self.y_batch:
            self.build_batches()

        # Prepare torch tensors
        hist_batch_Tensor = torch.FloatTensor(np.vstack(self.hist_batch))
        u_batch_Tensor = torch.FloatTensor(np.vstack(self.u_batch))
        self.y_batch_Tensor = torch.FloatTensor(np.vstack(self.y_batch))

        it = len(self.losses)
        for _ in range(train_it):
            print("iteration:", it)
            it += 1

            # Prepare
            phi_dyn = self.phi_fun(self.phi_c)
            self.y_hat_Tensor = self.model_dyn(u_batch_Tensor, hist_batch_Tensor, phi_dyn)
            mse = Functional.mse_loss(self.y_hat_Tensor, self.y_batch_Tensor, reduction='sum')
            print("MSE:", mse.item())
            phi_loss = mse
            if block_reg:
                block_reg = self.block_regularization(self.phi_c, error_weight=D_reg_weight)
                print("Block reg:", block_reg.item())
                phi_loss += block_reg
            if decay_reg:
                decay = self.decaying_impact_reguralization(self.phi_c, time_impact, error_weight=time_reg_weight)
                print("Decay reg:", decay.item())
                phi_loss += decay
            if conv_reg:
                convection = self.convection_impact_reguralization(self.phi_c, convection_impact,
                                                                   error_weight=conv_reg_weight)
                print("Convection reg:", convection.item())
                phi_loss += convection

            # Iterate
            self.phi_opt.zero_grad()
            phi_loss.backward()
            self.phi_opt.step()
            print(" Loss:", phi_loss.item())

            self.losses.append(phi_loss.item())

            if live_plt:
                self.live_plot()

            if not self.filename == "":
                self.save_dmpc(self.filename)
                if phi_loss.item() <= self.best_loss:  # Save the best iteration in it's own file
                    f = self.filename.split(".")
                    self.save_dmpc(f[0] + "_best." + f[1])
                    self.best_loss = phi_loss.item()

        if live_plt:
            plt.close()
            plt.show()

    def model_dyn(self, Acts, Hists, Phi):
        '''use current phi to make an estimate of measurements.
        Returns y_hat as a tensor'''
        Z = torch.cat((Hists, Acts), 1)
        y = torch.matmul(Z, Phi.T)
        return y

    def get_trajectory(self, batch):
        if self.p > 1:
            y = self.y_batch_Tensor[batch].detach().numpy()
            y = y.reshape(self.N, self.p).T
            y_hat = self.y_hat_Tensor[batch].detach().numpy()
            y_hat = y_hat.reshape(self.N, self.p).T
        else:
            y = self.y_batch_Tensor[batch].detach().numpy()
            y_hat = self.y_hat_Tensor[batch].detach().numpy()
        estimates = []
        true = []
        if self.p > 1:
            for i in range(self.p):
                estimates.append(y_hat[i, :].tolist())
                true.append(y[i, :].tolist())
        else:
            estimates = y_hat.tolist()
            true = y.tolist()
        return true, estimates

    # * Utility functions
    def save_dmpc(self, path):
        '''Save the class to a file'''
        file = open(path, 'wb')

        dmpc_dict = {
            'p': self.p,
            'm': self.m,
            'N': self.N,
            'H': self.H,
            'u': self.u,
            'y': self.y,
            'hist_batch': self.hist_batch,
            'u_batch': self.u_batch,
            'y_batch': self.y_batch,
            'phi_opt': self.phi_opt,
            'losses': self.losses,
            'best_loss': self.best_loss,
            'phi_c': self.phi_c,
            'lr': self.lr,
            'D_reg_weight': self.D_reg_weight,
            'time_reg_weight': self.time_reg_weight,
        }

        # pkl.dump(dmpc_dict, file, protocol=2)
        pkl.dump(dmpc_dict, file)
        file.close()

    def load_dmpc(self, path):
        ''' Load the class from a file'''
        try:
            with open(path, 'rb') as f:
                data = pkl.load(f)
            self.p = data['p']
            self.m = data['m']
            self.N = data['N']
            self.H = data['H']
            self.u = data['u']
            self.y = data['y']
            self.best_loss = data['best_loss']
            self.hist_batch = data['hist_batch']
            self.u_batch = data['u_batch']
            self.hist_batch = data['hist_batch']
            self.phi_opt = data['phi_opt']
            self.losses = data['losses']
            self.phi_c = data['phi_c']
            self.lr = data['lr']
            self.D_reg_weight = data['D_reg_weight']
            self.time_reg_weight = data['time_reg_weight']

        except:
            print("no such file: " + path)

    # TODO: can probably merge all phi_plot functions into one in some way, just vary input phi
    def phi_plot(self):
        '''Plot the phi matrix'''
        phi = self.phi_fun(self.phi_c)
        phi = phi.detach().numpy()
        plt.matshow(phi, vmin=np.amin(phi), vmax=np.amax(phi), cmap='viridis')
        plt.show(block=True)

    def phi_plot_block(self):
        '''used to plot diagonal matrices of phi'''
        self.find_block_indeces()
        phi_c_d = np.zeros(len(self.phi_c))
        for block in self.block_indeces:
            for i in block:
                phi_c_d[i] = 1000
        # plot phi matrix
        phi = self.phi_fun(phi_c_d, lib=np)
        plt.matshow(phi)
        plt.colorbar()
        plt.show(block=True)

    def phi_plot_decay(self):
        '''used to plot decaying time impact of phi'''

        phi_c_row = np.zeros(len(self.phi_c))

        time_impact = self.time_impact()
        index = 0
        print(time_impact.shape)
        print(time_impact.shape[1])
        print(time_impact.shape[0])
        for col in range(time_impact.shape[1]):
            for row in range(time_impact.shape[0]):
                phi_c_row[index] = time_impact[row, col]
                index += 1

        # plot phi matrix
        phi = self.phi_fun(phi_c_row, lib=np)
        plt.matshow(phi, vmin=np.amin(phi), vmax=np.amax(phi), cmap='viridis')
        plt.colorbar()
        plt.show(block=True)

    def phi_plot_convection(self):
        phi_conv = self.convection_impact()

        phi_conv = phi_conv.detach().numpy()
        numrows = phi_conv.shape[0]
        numcols = phi_conv.shape[1]
        plt.matshow(phi_conv, vmin=np.amin(phi_conv), vmax=np.amax(phi_conv), cmap='viridis',
                    extent=(0, numcols, 0, numrows))
        plt.show(block=True)

    def phi_plot_indices(self):
        # plot phi matrix where values of phi are indeces
        phi = self.phi_fun(np.arange(len(self.phi_c)), lib=np)
        plt.matshow(phi, vmin=np.amin(phi), vmax=np.amax(phi), cmap='viridis')
        plt.colorbar()
        plt.show(block=True)

    def live_plot_init(self):
        plt.ion()
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1)
        self.ax1.set_title("Phi")
        self.ax2.set_title("Loss")
        self.ax3.set_title("trajectories")
        # Phi plot
        phi = self.phi_fun(self.phi_c)
        phi = phi.detach().numpy()
        self.line1 = self.ax1.matshow(phi, vmin=np.amin(phi), vmax=np.amax(phi), cmap='viridis')
        plt.set_cmap('gist_ncar')
        plt.tight_layout()
        self.cbar = self.fig.colorbar(self.line1, ax=self.ax1)
        # Loss
        if len(self.losses) == 0:
            data = [0, 0, 0, 0, 0]
        else:
            data = self.losses
        self.line2, = self.ax2.plot(data)

        # Trajectory
        self.ax3.plot([0, 0, 0, 0, 0])

    def live_plot(self):
        phi = self.phi_fun(self.phi_c)
        phi = phi.detach().numpy()

        # Phi
        pos = self.ax1.matshow(phi, vmin=np.amin(phi), vmax=np.amax(phi), cmap='viridis')
        self.cbar.remove()
        self.cbar = self.fig.colorbar(pos, ax=self.ax1)
        # Losses
        self.ax2.clear()
        self.ax2.plot(self.losses)
        self.ax1.set_title("Phi")
        self.ax2.set_ylabel("Loss")
        self.ax2.set_xlabel("Iteration")
        n = len(self.losses)
        plt.tight_layout()
        # Scale losses plot
        if n >= self.losses_show_last:
            start = n - self.losses_show_last
            end = n - 1
            maxv = max(self.losses[start:end])
            self.ax2.set_ylim(min(self.losses) * 0.999, 1.001 * maxv)
            self.ax2.set_xlim(start, end)

        self.lowest_loss = self.ax2.axhline(y=min(self.losses), color='r', linestyle='--')
        # Trajectory
        n = randint(0, self.n_batches - 1)
        true, est = self.get_trajectory(n)
        self.ax3.clear()
        if self.p > 1:
            maxv = 0
            minv = 0
            for i in range(self.p):
                self.ax3.plot(true[i], label="y" + str(i))
                self.ax3.plot(est[i], ':', label="y" + str(i) + "_hat")
                maxv = max(maxv, max(max(true[i]), max(est[i])))
                minv = min(minv, min(min(true[i]), min(est[i])))
            self.ax3.legend()
        else:
            self.ax3.plot(true, label="y")
            self.ax3.plot(est, ':', label="y_hat")
            maxv = max(max(true), max(est))
            minv = min(min(true), min(est))
        self.ax3.set_ylim(min(0, minv * 0.9), maxv * 1.1)
        self.ax3.set_title("Batch number " + str(n))
        self.ax3.set_ylabel("Y")
        self.ax3.set_xlabel("step[k]")
        self.fig.canvas.flush_events()


def dataCheck(data):
    '''Checking that input data is valid'''
    if len(data) == 0:
        raise Exception("data is empty")
    if not (type(data) == list):
        raise Exception("data is not a list")