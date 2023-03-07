# standard libraries
import numpy as np
import casadi as csd
import torch.optim as optim
import pickle
import matplotlib.pyplot as plt
import pytz
import math
import art
import time
import os

import torch
from datetime import date, datetime, timedelta
import torch.nn.functional as F

# custom libraries and functions
from DMPC import DMPC

# * Load Data
print('Load Data')
f = open('DataHourly_Gap.pkl', "rb")
Data = pickle.load(f)
f.close()

# * House details
# Where are we?
Zone = 'Tr.heim'  # Spot Market : Trondheim
local_timezone = pytz.timezone('Europe/Oslo')  # Time zone   : Oslo

Pumps = ['main', 'living', 'livingdown', 'studio']
# Pumps = ['main','living']
observations = ['Temp']
actions = ['Fan', 'Out']

# * Setting up lists of all actions and observations; to be used for convection and other correlations
allAct = []
convAct = []
if 'Fan' in actions:
    for pump in Pumps:
        allAct.append(pump + '_Fan')
        convAct.append(pump + '_Fan')

if 'Out' in actions:
    allAct.append('Out')

allObs = []
convObs = []
if 'Temp' in observations:
    for pump in Pumps:
        allObs.append(pump + '_Temp')
        convObs.append(pump + '_Temp')

# parameters affected by convection
allIO = {'Act': allAct, 'Obs': allObs}
convIO = {'Act': convAct, 'Obs': convObs}

# * Parameters
horizon = 12  # 12 hour prediction
history = {'dt': 6, 'hours': 6}  # Depth (min is for dt-based data, hourly for hourly data --> weather)
dt = 5  # 5 minutes

maxHistory = 0
for item in history.keys():
    maxHistory = np.max([maxHistory, history[item]])

# * Data and Data setup
# Small data
Dates = {'Start': datetime(2022, 12, 1).astimezone(local_timezone),
         'End': datetime(2022, 12, 5).astimezone(local_timezone)}

# Big Data
"""Dates = {'Start': datetime(2021,5, 1).astimezone(local_timezone),
         'End'  : datetime(2022,5, 1).astimezone(local_timezone)}
"""

# Winter Data
"""Dates = {'Start': datetime(2021,11,  1).astimezone(local_timezone),
         'End'  : datetime(2022, 2, 28).astimezone(local_timezone)}
"""

# Small Data
"""Dates = {'Start': datetime(2021, 10,  1).astimezone(local_timezone),
         'End'  : datetime(2021, 10, 15).astimezone(local_timezone)}
"""

OutTemp = 'SEKLIMA'

# * Find valid indeces to build regressors:
Index_reg = []
Gap_indeces = []
for index, timeStamp in enumerate(Data['datetime']):
    if timeStamp >= Dates['Start'] and timeStamp < Dates['End']:
        Delta_T = (Data['datetime'][index + horizon - 1] - Data['datetime'][index - maxHistory]).total_seconds() / 60
        if Delta_T == (maxHistory + horizon - 1) * dt:  # check that we have valid points everywhere
            Index_reg.append(index)
        else:
            Gap_indeces.append((index, Data['datetime'][index]))

Ndata = len(Index_reg)
print('Model from ' + str(Dates['Start']))
print('...to ' + str(Dates['End']))
print('Prediction horizon: ' + str(horizon / 12) + 'h')
print('Data for regression : ' + str(Ndata))

# * Phi structure
"""                        u_ini                   
    y = [Phi_P | Phi_F] [  y_ini ]
                            u
"""

hist_batch = []
u_batch = []
y_batch = []

# make string versions of batches
hist_batch_str = []
u_batch_str = []
y_batch_str = []

nActions = 0
if 'Fan' in actions:
    nActions += len(Pumps)

if 'Out' in actions:
    nActions += 1

nObservations = 0
if 'Temp' in observations:
    nObservations += len(Pumps)

print(art.text2art('Data preparation and validity check in: ', font='doom'))

# countdown from 3 in 0.5s intervals with 0.5s pause using art package
for i in range(3, 0, -1):
    print(art.text2art(str(i) + '.', font='doom'))
    time.sleep(0.5)

print(art.text2art('GO!', font='doom'))

ValidIndeces = []  # list of indeces where all actions and observations are valid, i.e. not NaN
percentage = 0
for k, index in enumerate(Index_reg):

    # progress bar
    if k % int(len(Index_reg) / 20) == 0:
        print('[' + '=' * int(percentage / 5) + ' ' * (20 - int(percentage / 5)) + '] ' + str(percentage) + '%',
              flush=True, end='\r')
        percentage += 5

    pastActValid = True
    pastObsValid = True
    futureActValid = True
    futureObsValid = True

    pastIndeces = index - 1 - np.arange(
        maxHistory)  # -1 as to have the current index be part of the future. Based on Phi-strucutre
    pastIndeces = pastIndeces[::-1]  # reverse order
    futureIndeces = index + np.arange(horizon)

    # * History
    # Actions
    pastActions = np.zeros((nActions, maxHistory))
    pastActions_str = np.zeros((nActions, maxHistory), dtype=object)

    action = 0  # row index for action
    if 'Fan' in actions:
        for pump in Pumps:
            for k, pastInd in enumerate(pastIndeces):
                pump_action = Data[pump]['Set'][pastInd] - Data[pump]['Temp'][pastInd]
                pump_action = Data[pump]['On'][pastInd] * 0.5 * (np.tanh(0.2 * pump_action) + 1)

                fan_action = pump_action * Data[pump]['Fan'][pastInd]
                pastActions[action][k] = fan_action
                pastActions_str[action][k] = 'pastFan_' + pump + '_' + str(pastInd)

            action += 1

    if 'Out' in actions:
        for k, pastInd in enumerate(pastIndeces):
            pastActions[action][k] = Data[OutTemp][pastInd]
            pastActions_str[action][k] = 'pastOut_' + str(pastInd)
        action += 1

    # Reshape actions so that coloumns are actions by timestamp, i.e. [Fan1_-2 Fan2_-2 ; Fan1_-1 Fan2_-1 ; ...]
    pastActions = pastActions.T.reshape(-1)
    pastActions_str = pastActions_str.T.reshape(-1)

    # validity check
    if np.isnan(
            np.sum(pastActions)):  # fast check according to https://stackoverflow.com/questions/6736590/fast-check-for-nan-in-numpy
        pastActValid = False

    # Observations
    pastObservations = np.zeros((nObservations, maxHistory))
    pastObservations_str = np.zeros((nObservations, maxHistory), dtype=object)
    observation = 0  # row index for observation

    if 'Temp' in observations:
        for pump in Pumps:
            for k, pastInd in enumerate(pastIndeces):
                pastObservations[observation][k] = Data[pump]['Temp'][pastInd]
                pastObservations_str[observation][k] = 'pastTemp_' + pump + '_' + str(pastInd)
            observation += 1

    # Reshape observations so that coloumns are observations by timestamp, i.e. [Temp1_-2 Temp2_-2 ; Temp1_-1 Temp2_-1 ; ...]
    pastObservations = pastObservations.T.reshape(-1)
    pastObservations_str = pastObservations_str.T.reshape(-1)

    if np.isnan(np.sum(pastObservations)):
        pastObsValid = False

    # * Horizon
    # Actions
    futureActions = np.zeros((nActions, horizon))
    futureActions_str = np.zeros((nActions, horizon), dtype=object)

    action = 0  # row index for action
    if 'Fan' in actions:
        for pump in Pumps:
            for k, futureInd in enumerate(futureIndeces):
                pump_action = Data[pump]['Set'][futureInd] - Data[pump]['Temp'][futureInd]
                pump_action = Data[pump]['On'][futureInd] * 0.5 * (np.tanh(0.2 * pump_action) + 1)

                fan_action = pump_action * Data[pump]['Fan'][futureInd]
                futureActions[action][k] = fan_action
                futureActions_str[action][k] = 'futureFan_' + pump + '_' + str(futureInd)

            action += 1

    if 'Out' in actions:
        for k, futureInd in enumerate(futureIndeces):
            futureActions[action][k] = Data[OutTemp][futureInd]
            futureActions_str[action][k] = 'futureOut_' + str(futureInd)
        action += 1

    futureActions = futureActions.T.reshape(-1)
    futureActions_str = futureActions_str.T.reshape(-1)

    if np.isnan(np.sum(futureActions)):
        futureActValid = False

    # Observations
    futureObservations = np.zeros((nObservations, horizon))
    futureObservations_str = np.zeros((nObservations, horizon), dtype=object)
    observation = 0  # row index for observation

    if 'Temp' in observations:
        for pump in Pumps:
            for k, futureInd in enumerate(futureIndeces):
                futureObservations[observation][k] = Data[pump]['Temp'][futureInd]
                futureObservations_str[observation][k] = 'futureTemp_' + pump + '_' + str(futureInd)
            observation += 1

    futureObservations = futureObservations.T.reshape(-1)
    futureObservations_str = futureObservations_str.T.reshape(-1)

    if np.isnan(np.sum(futureObservations)):
        futureObsValid = False

    if pastActValid and pastObsValid and futureActValid and futureObsValid:
        # concatenate actions and observations, and append it to batch
        past = np.concatenate((pastActions, pastObservations))
        past_str = np.concatenate((pastActions_str, pastObservations_str))

        hist_batch.append(past)
        hist_batch_str.append(past_str)

        u_batch.append(futureActions)
        u_batch_str.append(futureActions_str)

        y_batch.append(futureObservations)
        y_batch_str.append(futureObservations_str)

        ValidIndeces.append(index)

print('Data for regression and validity check done, number of valid indeces: ' + str(len(ValidIndeces)))
print('Redueced number of indeces by ' + str(len(Index_reg) - len(ValidIndeces)), flush=True)

# * setting up torch
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# * set up DDMPC
print('-----....------')
# print 'Setting up DDMPC' in doom style using art package
print(art.text2art('Setting up DDMPC', font='doom'))
print('nObservations: ' + str(nObservations))
print('nActions: ' + str(nActions))
print('horizon: ' + str(horizon))
print('maxHistory: ' + str(maxHistory))
print('size of hist_batch: ' + str(len(hist_batch)))
print('size of u_batch: ' + str(len(u_batch)))
print('size of y_batch: ' + str(len(y_batch)))
print('all inputs and outputs: ' + str(allIO))
print('convection inputs and outputs ' + str(convIO))

dmpc = DMPC(p=nObservations, m=nActions, N=horizon, H=maxHistory, hist_batch=hist_batch, u_batch=u_batch,
            y_batch=y_batch, allIO=allIO, convIO=convIO)
dmpc.convection_impact()
dmpc.phi_plot_convection()
dmpc.calc_phi(train_it=1000, D_reg_weight=10000.0, time_reg_weight=100.0, conv_reg_weight=100.0, lr=0.01, live_plt=True,
              block_reg=True, decay_reg=True, conv_reg=True)
dmpc.phi_plot()

"""#! TEST SCENARIOS 
#*Test DMPC with different regularization weights, and learning rates
dmpc_testScenarios = []
#fill list with different combinations of regularization weights and learning rates
for lr in [0.01, 0.001, 0.0001]:
    for D_reg_weight in [10000.0, 1000.0, 100.0]:
        for time_reg_weight in [1000.0, 100.0, 100.0]:
            dmpc_testScenarios.append({'lr':lr, 'D_reg_weight':D_reg_weight, 'time_reg_weight':time_reg_weight})


print('-----....------')
#run DMPC with different regularization weights, and learning rates
testFolder = 'dmpcTests/'

if not os.path.exists(testFolder):
    os.makedirs(testFolder)

testIndex = 0
for test in dmpc_testScenarios:
    print('test scenario: ')
    print(test)
    testFileName = testFolder + 'test' + '-' + str(testIndex) + '.pkl'
    testIndex += 1

    dmpc_test = DMPC(p = nObservations, m=nActions, N=horizon, H=maxHistory, filename=testFileName, hist_batch=hist_batch, u_batch=u_batch, y_batch=y_batch, lr=test['lr'], D_reg_weight=test['D_reg_weight'], time_reg_weight=test['time_reg_weight'] )
    dmpc_test.calc_phi(train_it=700,lr=test['lr'], live_plt=False,block_reg=True,decay_reg=True, D_reg_weight=test['D_reg_weight'],time_reg_weight=test['time_reg_weight'])


#* load DMPC tests and find the best one by looking at the last element in the losses list in the class
dmpc_testFiles = os.listdir(testFolder)

#first one
dmpcBest = DMPC(p = nObservations, m=nActions, N=horizon, H=maxHistory, hist_batch=hist_batch, u_batch=u_batch, y_batch=y_batch)
dmpcBest.load_dmpc(testFolder+dmpc_testFiles[0])
bestDMPC = dmpc_testFiles[0]

for test in dmpc_testFiles[1:]:
    dmpc_test = DMPC()
    dmpc_test.load_dmpc(testFolder+test)

    if dmpc_test.best_loss < dmpcBest.best_loss:
        print('new best dmpc found: '+test)
        print('previous best loss: '+str(dmpcBest.best_loss))
        print('new best loss: '+str(dmpc_test.best_loss))
        print('setting new best DMPC')
        dmpcBest = dmpc_test
        bestDMPC = test   


print('-----....------')
print('best DMPC: '+bestDMPC)
print('best loss: '+str(dmpcBest.best_loss))
print('learning rate: '+str(dmpcBest.lr))
print('D_reg_weight: '+str(dmpcBest.D_reg_weight))
print('time_reg_weight: '+str(dmpcBest.time_reg_weight))
"""