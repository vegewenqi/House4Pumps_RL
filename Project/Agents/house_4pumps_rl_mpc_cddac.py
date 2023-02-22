from casadi.tools import *
import numpy as np
from Environments.house_4pumps import House4Pumps
from helpers import tqdm_context
from Agents.abstract_agent import TrainableController
from numpy import linalg as LA


class Custom_QP_formulation:
    def __init__(self, env, opt_horizon, gamma=1.0, th_param="custom", upper_tri=False):
        self.env = env
        self.pumps = self.env.pumps
        self.N = opt_horizon
        self.gamma = gamma
        self.etau = 1e-6
        self.th_param = th_param
        self.upper_tri = upper_tri

        ### build the nlp structure
        self.w = self.get_decision_variables()
        self.mpc_parameter = self.get_mpc_parameter_structure()
        self.lbG = None
        self.ubG = None
        self.X0 = self.w(0.01)
        self.dPi = None
        self.solver, self.dR_sensfunc = self.opt_formulation(self.w, self.mpc_parameter)

    def get_decision_variables(self):
        slacks = struct_symMX([entry('SlackDesTemp'),
                               entry('SlackMinTemp')])
        slacks_4pump = []
        for pump in self.pumps:
            slacks_4pump += [entry(pump, struct=slacks)]
        slacks_4pump = struct_symMX(slacks_4pump)

        ### Decision variables
        w = struct_symMX([
            entry('Input', struct=self.env.action_struct, repeat=self.N - 1),
            entry('State', struct=self.env.state_struct, repeat=self.N),
            entry('Slack', struct=slacks_4pump, repeat=self.N)
        ])

        return w

    def get_mpc_parameter_structure(self):
        state0_structure = self.env.state_struct

        ### data
        pump_1234 = []
        for pump in self.pumps:
            pump_1234 += [entry(pump)]
        pump_1234 = struct_symMX(pump_1234)

        data_structure = struct_symMX([entry('OutTemp', repeat=self.N),
                                       entry('Spot', repeat=self.N),
                                       entry('DesiredTemp', struct=pump_1234, repeat=self.N),
                                       entry('MinTemp', struct=pump_1234, repeat=self.N),
                                       entry('MaxFan', struct=pump_1234, repeat=self.N)
                                       ])

        ### theta
        theta_structure = struct_symMX([entry('TempAbove'),
                                        entry('TempBelow'),
                                        entry('MinTemp'),
                                        entry('Delta_TargetTemp'),
                                        entry('Delta_Fan'),
                                        entry('SpotGain'),
                                        # entry('PowerNoise'),
                                        entry('RoomNoise', struct=pump_1234),
                                        entry('HPset', struct=pump_1234),
                                        entry('HPin', struct=pump_1234),
                                        entry('HP0', struct=pump_1234),
                                        entry('Fan', struct=pump_1234),
                                        entry('m_air', struct=pump_1234),
                                        entry('m_wall', struct=pump_1234),
                                        entry('rho_in', struct=pump_1234),
                                        entry('rho_out', struct=pump_1234),
                                        entry('rho_dir', struct=pump_1234),
                                        ])

        ### mpc_parameter contains (state0, data, theta)
        mpc_parameter_structure = struct_symMX(
            [
                entry('state0', struct=state0_structure),
                entry('data', struct=data_structure),
                entry('theta', struct=theta_structure)
            ]
        )

        return mpc_parameter_structure

    def get_w_bound(self, mpc_parameter):
        lbw = self.w(-inf)
        ubw = self.w(inf)

        for pump in self.pumps:
            lbw['State', 0, pump, 'Room'] = mpc_parameter['state0', pump, 'Room']
            ubw['State', 0, pump, 'Room'] = mpc_parameter['state0', pump, 'Room']

            lbw['State', 0, pump, 'Wall'] = mpc_parameter['state0', pump, 'Wall']
            ubw['State', 0, pump, 'Wall'] = mpc_parameter['state0', pump, 'Wall']

            lbw['State', 0, pump, 'TargetTemp'] = mpc_parameter['state0', pump, 'TargetTemp']
            ubw['State', 0, pump, 'TargetTemp'] = mpc_parameter['state0', pump, 'TargetTemp']

            lbw['State', 0, pump, 'Fan'] = mpc_parameter['state0', pump, 'Fan']
            ubw['State', 0, pump, 'Fan'] = mpc_parameter['state0', pump, 'Fan']

        TargetTempLimit = {
            'main': {'Min': 10, 'Max': 31},
            'living': {'Min': 10, 'Max': 31},
            'studio': {'Min': 10, 'Max': 31},
            'livingdown': {'Min': 10, 'Max': 31}
        }

        for pump in self.pumps:
            lbw['State', 1:, pump, 'TargetTemp'] = TargetTempLimit[pump]['Min']
            ubw['State', 1:, pump, 'TargetTemp'] = TargetTempLimit[pump]['Max']

            lbw['State', 1:, pump, 'Fan'] = 0
            ubw['State', 1:, pump, 'Fan'] = mpc_parameter['data', 'MaxFan', :, pump]

            # lbw['Input', :, pump, 'On'] = 1
            # ubw['Input', :, pump, 'On'] = 1

            lbw['Slack', :, pump, 'SlackDesTemp'] = 0
            ubw['Slack', :, pump, 'SlackDesTemp'] = +inf

            lbw['Slack', :, pump, 'SlackMinTemp'] = 0
            ubw['Slack', :, pump, 'SlackMinTemp'] = +inf

        return lbw, ubw

    def opt_formulation(self, w, mpc_parameter):
        MaxPowGroup = {'East': {'pumps': ['main', 'studio', 'livingdown'], 'Power': 2},
                       'West': {'pumps': ['living'], 'Power': 1.5}}

        PowGroup = {}
        Cost24h = 0
        J = 0

        g = []
        h = []
        lbg = []
        ubg = []
        lbh = []
        ubh = []

        for k in range(self.N - 1):
            for key in MaxPowGroup.keys():
                PowGroup[key] = 0

            COP = self.env.COP(mpc_parameter['data', 'OutTemp', k])

            PowTot = 0

            for pump in self.pumps:
                ### power dynamics
                Pow = self.env.Pow(mpc_parameter['theta', 'HPset', pump],
                                   mpc_parameter['theta', 'HPin', pump],
                                   mpc_parameter['theta', 'HP0', pump],
                                   mpc_parameter['theta', 'Fan', pump],
                                   w['State', k, pump, 'TargetTemp'] + w['Input', k, pump, 'Delta_TargetTemp'],
                                   w['State', k, pump, 'Room'],
                                   w['State', k, pump, 'Fan'] + w['Input', k, pump, 'Delta_Fan'])
                # Pow *= w['Input', k, pump, 'On']

                h.append(0 - Pow)
                lbh.append(-inf)
                ubh.append(0)
                h.append(Pow - 1.5)
                lbh.append(-inf)
                ubh.append(0)

                for key in MaxPowGroup.keys():
                    if pump in MaxPowGroup[key]['pumps']:
                        PowGroup[key] += Pow

                PowTot += Pow

                ### State dynamics
                WallPlus = self.env.WallPlus(mpc_parameter['theta', 'm_wall', pump],
                                             mpc_parameter['theta', 'rho_out', pump],
                                             mpc_parameter['theta', 'rho_in', pump],
                                             w['State', k, pump, 'Wall'],
                                             w['State', k, pump, 'Room'],
                                             mpc_parameter['data', 'OutTemp', k])

                RoomPlus = self.env.RoomPlus(mpc_parameter['theta', 'm_air', pump],
                                             mpc_parameter['theta', 'rho_in', pump],
                                             mpc_parameter['theta', 'rho_dir', pump],
                                             w['State', k, pump, 'Room'],
                                             w['State', k, pump, 'Wall'],
                                             mpc_parameter['data', 'OutTemp', k],
                                             COP,
                                             Pow)

                RoomPlus += mpc_parameter['theta', 'RoomNoise', pump]

                g.append(RoomPlus - mpc_parameter['theta', 'm_air', pump] * w['State', k + 1, pump, 'Room'])
                lbg.append(0)
                ubg.append(0)

                g.append(WallPlus - mpc_parameter['theta', 'm_wall', pump] * w['State', k + 1, pump, 'Wall'])
                lbg.append(0)
                ubg.append(0)

                ### Control dynamics
                g.append(w['State', k, pump, 'TargetTemp'] + w['Input', k, pump, 'Delta_TargetTemp'] - w[
                    'State', k + 1, pump, 'TargetTemp'])
                lbg.append(0)
                ubg.append(0)

                g.append(w['State', k, pump, 'Fan'] + w['Input', k, pump, 'Delta_Fan'] - w[
                    'State', k + 1, pump, 'Fan'])
                lbg.append(0)
                ubg.append(0)

                ### Slack stuff
                h.append(mpc_parameter['data', 'DesiredTemp', k, pump] - w['State', k, pump, 'Room'] - w[
                    'Slack', k, pump, 'SlackDesTemp'])
                lbh.append(-inf)
                ubh.append(0)

                h.append(mpc_parameter['data', 'MinTemp', k, pump] - w['State', k, pump, 'Room'] - w[
                    'Slack', k, pump, 'SlackMinTemp'])
                lbh.append(-inf)
                ubh.append(0)

                ### Cost
                J += mpc_parameter['theta', 'TempAbove'] * (mpc_parameter['data', 'DesiredTemp', k, pump] - w[
                    'State', k, pump, 'Room']) ** 2 / float(self.N)

                J += mpc_parameter['theta', 'TempBelow'] * w['Slack', k, pump, 'SlackDesTemp'] ** 2 / float(self.N)
                J += mpc_parameter['theta', 'MinTemp'] * w['Slack', k, pump, 'SlackMinTemp'] ** 2 / float(self.N)

                # J += 10**(-16) * mpc_parameter['theta', 'Delta_TargetTemp'] * w[
                #     'Input', k, pump, 'Delta_TargetTemp'] ** 2 / float(self.N)
                # J += 10**(-16) * mpc_parameter['theta', 'Delta_Fan'] * w[
                #     'Input', k, pump, 'Delta_Fan'] ** 2 / float(self.N)

                J += mpc_parameter['theta', 'Delta_TargetTemp'] * w[
                    'Input', k, pump, 'Delta_TargetTemp'] ** 2 / float(self.N)
                J += mpc_parameter['theta', 'Delta_Fan'] * w['Input', k, pump, 'Delta_Fan'] ** 2 / float(self.N)

            J += mpc_parameter['theta', 'SpotGain'] * (mpc_parameter['data', 'Spot', k] + self.env.GridCost
                                                       ) * PowTot / float(self.N)
            Cost24h += (24 / self.env.Hour_MPC_Horizon) * (mpc_parameter['data', 'Spot', k] + self.env.GridCost
                                                           ) * PowTot * self.env.HPSamplingTime / 60 / 100

            for key in MaxPowGroup.keys():
                h.append(PowGroup[key] - MaxPowGroup[key]['Power'])
                lbh.append(-inf)
                ubh.append(0)

        ### Terminal constraints and costs
        for pump in self.pumps:
            h.append(mpc_parameter['data', 'DesiredTemp', -1, pump] - w['State', -1, pump, 'Room'] - w[
                'Slack', -1, pump, 'SlackDesTemp'])
            lbh.append(-inf)
            ubh.append(0)

            h.append(mpc_parameter['data', 'MinTemp', -1, pump] - w['State', -1, pump, 'Room'] - w[
                'Slack', -1, pump, 'SlackMinTemp'])
            lbh.append(-inf)
            ubh.append(0)

            J += mpc_parameter['theta', 'TempAbove'] * (
                    mpc_parameter['data', 'DesiredTemp', -1, pump] - w['State', -1, pump, 'Room']) ** 2 / float(self.N)
            J += mpc_parameter['theta', 'TempBelow'] * w['Slack', -1, pump, 'SlackDesTemp'] ** 2 / float(self.N)
            J += mpc_parameter['theta', 'MinTemp'] * w['Slack', -1, pump, 'SlackMinTemp'] ** 2 / float(self.N)

        ### Maximum 24h cost
        Max24hCost = 100
        h.append(Cost24h - Max24hCost)
        lbh.append(-inf)
        ubh.append(0)

        g = vertcat(*g)
        h = vertcat(*h)
        G = vertcat(g, h)
        self.lbG = vertcat(*lbg, *lbh)
        self.ubG = vertcat(*ubg, *ubh)

        # NLP Problem for value function and policy approximation
        opts_setting = {
            "ipopt.max_iter": 500,
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.mu_target": self.etau,
            "ipopt.mu_init": self.etau,
            "ipopt.acceptable_tol": 1e-5,
            "ipopt.acceptable_obj_change_tol": 1e-5,
        }

        nlp_prob = {
            "f": J,
            "x": self.w,
            "p": self.mpc_parameter,
            "g": G
        }

        solver = nlpsol("solver", "ipopt", nlp_prob, opts_setting)

        dR_sensfunc = self.build_sensitivity(J, g, h)
        # dR_sensfunc = None  ### when mode="mpc"

        return solver, dR_sensfunc

    def build_sensitivity(self, J, g, h):
        lamb = MX.sym("lamb", g.shape[0])
        mu = MX.sym("mu", h.shape[0])
        mult = vertcat(lamb, mu)

        Lag = J + transpose(lamb) @ g + transpose(mu) @ h

        Lagfunc = Function("Lag", [self.w, mult, self.mpc_parameter], [Lag])
        dLagfunc = Lagfunc.factory("dLagfunc", ["i0", "i1", "i2"], ["jac:o0:i0"])
        dLdw = dLagfunc(self.w, mult, self.mpc_parameter)
        Rr = vertcat(transpose(dLdw), g, mu * h + self.etau)
        z = vertcat(self.w, mult)
        R_kkt = Function("R_kkt", [z, self.mpc_parameter], [Rr])
        dR_sensfunc = R_kkt.factory("dR", ["i0", "i1"], ["jac:o0:i0", "jac:o0:i1"])

        return dR_sensfunc


class Custom_MPCActor(Custom_QP_formulation):
    def __init__(self, env, mpc_horizon, cost_params, gamma=1.0, debug=False):
        upper_tri = cost_params["upper_tri"] if "upper_tri" in cost_params else False
        super().__init__(env, mpc_horizon, gamma, cost_params["cost_defn"], upper_tri)
        self.debug = debug
        self.dim_state0 = self.mpc_parameter['state0'].shape[0]
        self.dim_dataN = self.mpc_parameter['data'].shape[0]
        self.dim_input0 = self.w['Input', 0].shape[0]

        self.mpc_parameter_value = self.mpc_parameter(0)

        self.actor_wt_structure = self.mpc_parameter.struct.dict['theta'].dict['struct']
        self.actor_wt = self.actor_wt_structure(0)

        self.actor_wt['TempAbove'] = self.env.config['cost_weights']['TempAbove']
        self.actor_wt['TempBelow'] = self.env.config['cost_weights']['TempBelow']
        self.actor_wt['MinTemp'] = self.env.config['cost_weights']['MinTemp']
        self.actor_wt['Delta_TargetTemp'] = self.env.config['cost_weights']['Delta_TargetTemp']
        self.actor_wt['Delta_Fan'] = self.env.config['cost_weights']['Delta_Fan']
        self.actor_wt['SpotGain'] = self.env.config['cost_weights']['SpotGain']
        # self.actor_wt['PowerNoise'] = cost_params['theta_power_noise']
        self.actor_wt['RoomNoise'] = cost_params['theta_room_noise']
        for pump in self.pumps:
            self.actor_wt['HPset', pump] = self.env.params['HPset'][pump],
            self.actor_wt['HPin', pump] = self.env.params['HPin'][pump],
            self.actor_wt['HP0', pump] = self.env.params['HP0'][pump]
            self.actor_wt['Fan', pump] = self.env.params['Fan'][pump],
            self.actor_wt['m_air', pump] = self.env.params['m_air'][pump],
            self.actor_wt['m_wall', pump] = self.env.params['m_wall'][pump],
            self.actor_wt['rho_in', pump] = self.env.params['rho_in'][pump],
            self.actor_wt['rho_out', pump] = self.env.params['rho_out'][pump],
            self.actor_wt['rho_dir', pump] = self.env.params['rho_dir'][pump],

        self.soln = None
        self.info = None

    def act_forward(self, state: DM, act_wt=None, time=None, mode="train"):
        act_wt = act_wt if act_wt is not None else self.actor_wt
        time = time if time is not None else self.env.t

        for pump in self.pumps:
            self.mpc_parameter_value['state0', pump, 'Room'] = state[pump, 'Room']
            self.mpc_parameter_value['state0', pump, 'Wall'] = state[pump, 'Wall']
            self.mpc_parameter_value['state0', pump, 'TargetTemp'] = state[pump, 'TargetTemp']
            self.mpc_parameter_value['state0', pump, 'Fan'] = state[pump, 'Fan']

        self.mpc_parameter_value['data', 'OutTemp'] = self.env.data['OutTemp'][time:time+self.N]
        self.mpc_parameter_value['data', 'Spot'] = self.env.data['Spot'][time:time+self.N]
        for pump in self.pumps:
            self.mpc_parameter_value['data', 'DesiredTemp', :, pump] = self.env.data['DesiredTemp'][
                                                                           pump][time:time+self.N]
            self.mpc_parameter_value['data', 'MinTemp', :, pump] = self.env.data['MinTemp'][
                                                                       pump][time:time+self.N]
            self.mpc_parameter_value['data', 'MaxFan', :, pump] = self.env.data['MaxFan'][
                                                                      pump][time:time+self.N]

        self.mpc_parameter_value['theta'] = act_wt

        lbw, ubw = self.get_w_bound(self.mpc_parameter_value)

        self.soln = self.solver(x0=self.X0,
                                lbx=lbw,
                                ubx=ubw,
                                lbg=self.lbG,
                                ubg=self.ubG,
                                p=self.mpc_parameter_value)

        flag = self.solver.stats()
        if not flag["success"]:
            RuntimeError("Problem is Infeasible")

        opt_var = self.soln["x"].full().flatten()
        opt_var = self.w(opt_var)

        # ### Compute MPC open-loop result for debug
        # MPCPow = {}
        # PowFac = 1
        # cost_24h = 0
        # cost_power = 0
        # cost_deviation = 0
        # cost_slack = 0
        # cost_slackMin = 0
        # cost_delTemp = 0
        # cost_delFan = 0
        # PowerTot_mpc = 0
        # for pump in self.pumps:
        #     MPCPow[pump] = []
        #     for k in range(self.N-1):
        #         Pow = self.env.Pow(self.mpc_parameter_value['theta', 'HPset', pump],
        #                            self.mpc_parameter_value['theta', 'HPin', pump],
        #                            self.mpc_parameter_value['theta', 'HP0', pump],
        #                            self.mpc_parameter_value['theta', 'Fan', pump],
        #                            opt_var['State', k, pump, 'TargetTemp'] + opt_var[
        #                                'Input', k, pump, 'Delta_TargetTemp'],
        #                            opt_var['State', k, pump, 'Room'],
        #                            opt_var['State', k, pump, 'Fan'] + opt_var['Input', k, pump, 'Delta_Fan'])
        #         # Pow *= w_opt['Input', k, pump, 'On']
        #         MPCPow[pump].append(PowFac * Pow)
        #         PowerTot_mpc += Pow

        #     cost_24h += self.cost_24h(np.array(self.mpc_parameter_value['data', 'Spot', :-1]).squeeze(),
        #                               np.array(MPCPow[pump][:]).squeeze())

        #     cost_power += self.cost_power(self.mpc_parameter_value['theta', 'SpotGain'],
        #                                   np.array(self.mpc_parameter_value['data', 'Spot', :-1]).squeeze(),
        #                                   0,
        #                                   np.array(MPCPow[pump][:]).squeeze())

        #     cost_deviation += self.cost_deviation(self.mpc_parameter_value['theta', 'TempAbove'],
        #                                           np.array(self.mpc_parameter_value[
        #                                                        'data', 'DesiredTemp', :, pump]).squeeze(),
        #                                           np.array(opt_var['State', :, pump, 'Room']).squeeze())

        #     cost_slack += self.cost_slack2_delta2(self.mpc_parameter_value['theta', 'TempBelow'],
        #                                           np.array(opt_var['Slack', :, pump, 'SlackDesTemp']).squeeze())

        #     cost_slackMin += self.cost_slack2_delta2(self.mpc_parameter_value['theta', 'MinTemp'],
        #                                              np.array(opt_var['Slack', :, pump, 'SlackMinTemp']).squeeze())

        #     cost_delTemp += self.cost_slack2_delta2(self.mpc_parameter_value['theta', 'Delta_TargetTemp'],
        #                                             np.array(opt_var['Input', :, pump, 'Delta_TargetTemp']).squeeze())

        #     cost_delFan += self.cost_slack2_delta2(self.mpc_parameter_value['theta', 'Delta_Fan'],
        #                                            np.array(opt_var['Input', :, pump, 'Delta_Fan']).squeeze())

        # mpc_cost_sum = cost_deviation + cost_slack + cost_slackMin + cost_delTemp + cost_delFan + cost_power
        # print('cost_24h = %f' % cost_24h)
        # print('cost_power = %f' % cost_power)
        # print('cost_deviation = %f' % cost_deviation)
        # print('cost_slack = %f' % cost_slack)
        # print('cost_slackMin = %f' % cost_slackMin)
        # print('cost_delTemp = %f' % cost_delTemp)
        # print('cost_delFan = %f' % cost_delFan)
        # print('mpc_cost_f - mpc_cost_sum = %f' % (self.soln['f'] - mpc_cost_sum))
        # print('mpc_cost_sum=%f' % mpc_cost_sum)
        # print('mpc_cost_f=%f' % self.soln['f'])
        # print(f'PowerTot_mpc={PowerTot_mpc}')

        # ### reset self.X0 for next calculation
        # self.X0 = opt_var

        act = self.env.action_struct(0)
        for pump in self.pumps:
            act[pump, 'Delta_TargetTemp'] = opt_var['Input', 0, pump, 'Delta_TargetTemp']
            act[pump, 'Delta_Fan'] = opt_var['Input', 0, pump, 'Delta_Fan']
            # act[pump, 'On'] = opt_var['Input', 0, pump, 'On']

        ### add time info as additional infos
        self.info = {"soln": self.soln, "time": time, "act_wt": act_wt}

        if self.debug:
            print("Soln:")
            print(opt_var['State'])
            print(opt_var['Input'])
            print(opt_var['Slack'])

        return act, self.info

    def dPidP(self, state: DM, act_wt: DM, info):
        soln = info["soln"]
        x = soln["x"].full()
        lam_g = soln["lam_g"].full()
        z = np.concatenate((x, lam_g), axis=0)

        for pump in self.pumps:
            self.mpc_parameter_value['state0', pump, 'Room'] = state[pump, 'Room']
            self.mpc_parameter_value['state0', pump, 'Wall'] = state[pump, 'Wall']
            self.mpc_parameter_value['state0', pump, 'TargetTemp'] = state[pump, 'TargetTemp']
            self.mpc_parameter_value['state0', pump, 'Fan'] = state[pump, 'Fan']

        self.mpc_parameter_value['theta'] = act_wt

        time = info["time"]
        self.mpc_parameter_value['data', 'OutTemp'] = self.env.data['OutTemp'][time:time + self.N]
        self.mpc_parameter_value['data', 'Spot'] = self.env.data['Spot'][time:time + self.N]
        for pump in self.pumps:
            self.mpc_parameter_value['data', 'DesiredTemp', :, pump] = self.env.data['DesiredTemp'][
                                                                           pump][time:time + self.N]
            self.mpc_parameter_value['data', 'MinTemp', :, pump] = self.env.data['MinTemp'][
                                                                       pump][time:time + self.N]
            self.mpc_parameter_value['data', 'MaxFan', :, pump] = self.env.data['MaxFan'][
                                                                      pump][time:time + self.N]

        [dRdz, dRdP] = self.dR_sensfunc(z, self.mpc_parameter_value)
        dzdP = (-np.linalg.solve(dRdz, dRdP[:, self.dim_state0 + self.dim_dataN:])).T
        # dzdP1 = np.array((-inv(dRdz) @ dRdP[:, dim_state0+dim_data:]).T)
        ### super slow with inv(), and the results of this two are different
        dpi = dzdP[:, :self.dim_input0]

        return dpi

    ### functions for calculating the MPC costs
    def param_update(self, lr, dJ, act_wt):
        print("Not implemented: constrained param update")
        return act_wt

    ### functions for calculating the MPC costs
    def cost_deviation(self, data_TempAbove, data_DeiredTemp, state_RoomTemp):
        cost_deviation = data_TempAbove * LA.norm((data_DeiredTemp - state_RoomTemp), ord=2) ** 2 / float(self.N)
        return cost_deviation

    def cost_slack2_delta2(self, data, state):
        cost = data * LA.norm(state, ord=2) ** 2 / float(self.N)
        return cost

    def cost_power(self, data_Gain, data_Spot, data_Base, pow):
        cost_power = data_Gain * np.dot(np.squeeze(data_Spot + self.env.GridCost - data_Base), pow) / float(self.N)
        return cost_power

    def cost_24h(self, data_Spot, pow):
        cost_24h = (24 / self.env.Hour_MPC_Horizon) * np.dot((data_Spot + self.env.GridCost),
                                                             pow) * self.env.HPSamplingTime / 60 / 100
        return cost_24h


class House_4Pumps_MPCAgent(TrainableController):
    def __init__(self, env, agent_params):
        super(House_4Pumps_MPCAgent, self).__init__(env=env)
        self.env = env
        self.pumps = self.env.pumps

        ### Hyper-parameters
        self._parse_agent_params(**agent_params)

        ### Actor initialization
        self.mpc_horizon_N = int(self.env.Hour_MPC_Horizon * 60 / self.env.HPSamplingTime)
        self.actor = Custom_MPCActor(self.env, self.mpc_horizon_N, self.cost_params, self.gamma, self.debug)

        self.obs_dim = self.env.state_struct.shape[0]
        self.action_dim = self.env.action_struct.shape[0]

        ### Critic params initialization
        # dim_sfeature = int(2 * self.obs_dim + 1)
        dim_sfeature = int((self.obs_dim + 2)*(self.obs_dim + 1)/2)  ### if consider cross terms, use this formula
        self.vi = 0.00001 * np.random.randn(dim_sfeature, 1)
        self.omega = 0.00001 * np.random.randn(self.actor.actor_wt.shape[0], 1)
        self.nu = 0.00001 * np.random.randn(self.actor.actor_wt.shape[0], 1)
        self.adam_m = 0
        self.adam_n = 0

        self.num_policy_update = 0

        self.theta_bound = np.concatenate((np.zeros(6), -np.ones(4)*np.inf, np.zeros(36)))

        ### Render prep
        self.fig = None
        self.ax = None

    def _parse_agent_params(self, cost_params, eps, gamma, actor_lr, nu_lr, vi_lr, omega_lr, policy_delay, debug,
                            train_params):
        self.cost_params = cost_params
        self.eps = eps
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.nu_lr = nu_lr
        self.vi_lr = vi_lr
        self.omega_lr = omega_lr
        self.policy_delay = policy_delay
        self.debug = debug
        self.iterations = train_params["iterations"]
        self.batch_size = train_params["batch_size"]

    def get_action(self, state: DM, act_wt=None, time=None, mode="train"):
        eps = self.eps if mode == "train" else 0.0

        act, info = self.actor.act_forward(state, act_wt=act_wt, time=time, mode=mode)

        act = self.env.action_struct(act.cat +
                                     eps * (-0.5 + np.random.rand(self.action_dim)) * [8, 1, 8, 1, 8, 1, 8, 1])

        for pump in self.pumps:
            act[pump, 'Delta_TargetTemp'] = np.clip(act[pump, 'Delta_TargetTemp'],
                                                    self.env.action_space[pump]['Delta_TargetTemp']['low'],
                                                    self.env.action_space[pump]['Delta_TargetTemp']['high'])
            act[pump, 'Delta_Fan'] = np.clip(act[pump, 'Delta_Fan'],
                                             self.env.action_space[pump]['Delta_Fan']['low'],
                                             self.env.action_space[pump]['Delta_Fan']['high'])
            # act[pump, 'On'] = np.clip(act[pump, 'On'],
            #                           self.env.action_space[pump]['On']['low'],
            #                           self.env.action_space[pump]['On']['high'])
        return act, info

    def state_to_feature(self, state):
        SS = np.triu(np.outer(state, state))
        size = state.shape[0]
        phi_s = []
        for row in range(size):
            for col in range(row, size):
                phi_s.append(SS[row][col])
        phi_s = np.concatenate((phi_s, state, 1.0), axis=None)[:, None]
        return phi_s

    def get_V_value(self, phi_s):
        V = np.matmul(phi_s.T, self.vi)
        return V

    def state_action_to_feature(self, action, pi, dpi_dtheta):
        phi_sa = np.matmul(dpi_dtheta, (action-pi))
        return phi_sa

    def get_Q_value(self, phi_sa, V):
        Q = np.matmul(phi_sa.T, self.omega) + V
        return Q

    def train(self, replay_buffer, train_it):
        delta_dpidpi = 0
        for train_i in tqdm_context(range(train_it), desc="Training Iterations"):
            print(f'batch training iteration {train_i}')
            states, actions, rewards, next_states, dones, infos = replay_buffer.sample(self.batch_size)
            delta_nu = 0
            delta_vi = 0
            delta_omega = 0
            for j, s in enumerate(states):
                # all info need for (s,a)
                s_array = s.cat.full().squeeze()
                phi_s = self.state_to_feature(s_array)
                V_s = self.get_V_value(phi_s)
                info_s = infos[j]
                pi_s = info_s["soln"]["x"].full()[: self.action_dim]
                action_s = actions[j].cat.full()
                dpi_dtheta_s = self.actor.dPidP(s, info_s["act_wt"], info_s)
                phi_sa = self.state_action_to_feature(action_s, pi_s, dpi_dtheta_s)
                Q_sa = self.get_Q_value(phi_sa, V_s)

                # all info need for (s', pi_s')
                ns = next_states[j]
                ns_array = ns.cat.full().squeeze()
                phi_ns = self.state_to_feature(ns_array)
                V_ns = self.get_V_value(phi_ns)
                # action_ns, info_ns = self.get_action(ns, act_wt=self.actor.actor_wt, time=info_s["time"]+1)
                # pi_ns = info_ns["soln"]["x"].full()[: self.action_dim]
                # dpi_dtheta_ns = self.actor.dPidP(ns, info_ns["act_wt"], info_ns)
                # phi_nsna = self.state_action_to_feature(action_s, pi_s, dpi_dtheta_s)

                td_error = rewards[j] + self.gamma * V_ns - Q_sa
                delta_nu += ((td_error - np.matmul(phi_sa.T, self.nu)) * phi_sa) / self.batch_size
                delta_vi += (td_error * phi_s - self.gamma * np.matmul(phi_sa.T, self.nu) * phi_ns) / self.batch_size
                delta_omega += (td_error * phi_sa) / self.batch_size
                # delta_vi += (td_error * phi_s) / self.batch_size
                # delta_omega += (td_error * phi_sa) / self.batch_size
                delta_dpidpi += np.matmul(dpi_dtheta_s, dpi_dtheta_s.T)

            # delta_nu = delta_nu / self.batch_size
            # delta_vi = delta_vi / self.batch_size
            # delta_omega = delta_omega / self.batch_size

            print(f'update critic')
            self.nu = self.nu + self.nu_lr * delta_nu
            self.vi = self.vi + self.vi_lr * delta_vi
            self.omega = self.omega + self.omega_lr * delta_omega
            # print(f'self.nu = {self.nu.squeeze()}')
            print(f'td_error = {td_error}')
            # print(f'delta_vi = {delta_vi.squeeze()}')
            # print(f'self.vi = {self.vi.squeeze()}')
            # print(f'self.omega = {self.omega.squeeze()}')

            if (train_i+1) % self.policy_delay == 0:
                print(f'update actor')
                dpi_dpidpi_avg = delta_dpidpi / ((train_i+1) * self.batch_size)
                delta_theta = np.matmul(dpi_dpidpi_avg, self.omega)
                delta_dpidpi = 0
                self.num_policy_update += 1
                # Adam
                self.adam_m = 0.9 * self.adam_m + (1-0.9) * delta_theta
                m_hat = self.adam_m / (1-0.9**self.num_policy_update)
                self.adam_n = 0.999 * self.adam_n + (1-0.999) * delta_theta**2
                n_hat = self.adam_n / (1-0.999**self.num_policy_update)
                new_theta = self.actor.actor_wt.cat.full() - self.actor_lr * (m_hat / (np.sqrt(n_hat)+10**(-8)))
                constrained_new_theta = np.maximum(new_theta.squeeze(), self.theta_bound)
                self.actor.actor_wt = self.actor.actor_wt_structure(constrained_new_theta)
                print(f'self.actor.actor_wt = {self.actor.actor_wt.cat}')
        # print('hi')


### test
if __name__ == "__main__":
    import numpy as np
    from casadi.tools import *
    import os
    import json
    from replay_buffer import BasicBuffer

    json_path = os.path.abspath(os.path.join(os.getcwd(), '../../Settings/other/house_4pumps_rl_mpc_cddac.json'))
    with open(json_path, 'r') as f:
        params = json.load(f)
        params["env_params"]["json_path"] = json_path
        print(f'env_params = {params}')
    env = House4Pumps(params["env_params"])

    # ### test Custom_QP_formulation()
    # a = Custom_QP_formulation(env, 6)

    # ## test Custom_MPCActor()
    # b = Custom_MPCActor(env, 6, params["agent_params"]["cost_params"], debug=True)
    # act, info = b.act_forward(env.reset())
    # jacob_act = b.dPidP(env.reset(), b.actor_wt, info)
    # print(jacob_act.shape)

    ### test Smart_Home_MPCAgent
    c = House_4Pumps_MPCAgent(env, params["agent_params"])