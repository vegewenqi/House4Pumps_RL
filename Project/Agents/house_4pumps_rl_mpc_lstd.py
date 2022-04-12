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
        self.X0 = self.w(0)
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

    def act_forward(self, state: DM, act_wt=None, mode="train"):
        act_wt = act_wt if act_wt is not None else self.actor_wt

        for pump in self.pumps:
            self.mpc_parameter_value['state0', pump, 'Room'] = state[pump, 'Room']
            self.mpc_parameter_value['state0', pump, 'Wall'] = state[pump, 'Wall']
            self.mpc_parameter_value['state0', pump, 'TargetTemp'] = state[pump, 'TargetTemp']
            self.mpc_parameter_value['state0', pump, 'Fan'] = state[pump, 'Fan']

        self.mpc_parameter_value['data', 'OutTemp'] = self.env.data['OutTemp'][self.env.t:self.env.t + self.N]
        self.mpc_parameter_value['data', 'Spot'] = self.env.data['Spot'][self.env.t:self.env.t + self.N]
        for pump in self.pumps:
            self.mpc_parameter_value['data', 'DesiredTemp', :, pump] = self.env.data['DesiredTemp'][
                                                                           pump][self.env.t:self.env.t + self.N]
            self.mpc_parameter_value['data', 'MinTemp', :, pump] = self.env.data['MinTemp'][
                                                                       pump][self.env.t:self.env.t + self.N]
            self.mpc_parameter_value['data', 'MaxFan', :, pump] = self.env.data['MaxFan'][
                                                                      pump][self.env.t:self.env.t + self.N]

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
        self.info = {"soln": self.soln, "time": self.env.t, "act_wt": act_wt}

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

        ### Critic params
        dim_sfeature = int(2 * self.obs_dim + 1)
        ### dim_sfeature = (self.obs_dim + 2)*(self.obs_dim + 1)/2  ### if consider cross terms, use this formula
        self.critic_wt = 0.01 * np.random.rand(dim_sfeature, 1)
        self.adv_wt = 0.01 * np.random.rand(self.actor.actor_wt.shape[0], 1)

        ### Render prep
        self.fig = None
        self.ax = None

    def state_to_feature(self, state):
        S = np.diag(np.outer(state, state))
        S = np.concatenate((S, state, np.array([1.0])))
        return S

    def get_value(self, state):
        S = self.state_to_feature(state)
        V = S.dot(self.critic_wt)
        return V

    def get_action(self, state: DM, act_wt=None, mode="train"):
        eps = self.eps if mode == "train" else 0.0

        act, info = self.actor.act_forward(state, act_wt=act_wt, mode=mode)

        act = self.env.action_struct(act.cat + eps * np.random.rand(self.action_dim) * [8, 1, 8, 1, 8, 1, 8, 1])

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

    def train(self, replay_buffer, train_it):
        # for critic param update
        Av = np.zeros(shape=(self.critic_wt.shape[0], self.critic_wt.shape[0]))
        bv = np.zeros(shape=(self.critic_wt.shape[0], 1))
        # for advantage fn param
        Aq = np.zeros(shape=(self.adv_wt.shape[0], self.adv_wt.shape[0]))
        bq = np.zeros(shape=(self.adv_wt.shape[0], 1))
        G = np.zeros(shape=(self.adv_wt.shape[0], self.adv_wt.shape[0]))

        for _ in tqdm_context(range(train_it), desc="Training Iterations"):
            states, actions, rewards, next_states, dones, infos = replay_buffer.sample(self.batch_size)
            for j, s in enumerate(states):
                s_array = s.cat.full().squeeze()
                S = self.state_to_feature(np.array(s_array).squeeze())[:, None]
                next_states_j_array = next_states[j].cat.full().squeeze()
                temp = S - (1 - dones[j]) * self.gamma * self.state_to_feature(next_states_j_array)[:, None]
                Av += np.matmul(S, temp.T)
                bv += rewards[j] * S

                if self.experience_replay:  # use current theta to generate new data
                    self.env.t = infos[j]["time"]
                    pi_act, info = self.get_action(s, self.actor.actor_wt, mode="update")
                    pi_act = pi_act.cat.full()
                    jacob_pi = self.actor.dPidP(s, self.actor.actor_wt, info)
                    actions[j] = info["soln"]["x"].full()[: self.action_dim]
                else:  # use old data
                    info = infos[j]
                    soln = info["soln"]
                    pi_act = soln["x"].full()[: self.action_dim]
                    jacob_pi = self.actor.dPidP(s, info["act_wt"], info)
                    actions[j] = actions[j].cat.full()

                    ### rescale jacob_pi
                    # jacob_pi[3, :] = jacob_pi[3, :] * 10**(-16)
                    # jacob_pi[4, :] = jacob_pi[4, :] * 10**(-16)
                    jacob_pi[0:3, :] = jacob_pi[0:3, :] * 10 ** 16
                    jacob_pi[5:, :] = jacob_pi[5:, :] * 10 ** 16

                psi = np.matmul(jacob_pi, (actions[j] - pi_act))

                Aq += np.matmul(psi, psi.T)
                bq += psi * (rewards[j] + (1 - dones[j]) * self.gamma * self.get_value(
                    next_states_j_array) - self.get_value(s_array))

                G += np.matmul(jacob_pi, jacob_pi.T)

        self.critic_wt = np.linalg.solve(Av, bv)
        # print(f'self.critic_wt = {self.critic_wt}')

        ### debug for Aq
        # aa = np.linalg.det(Aq)
        # bb = np.linalg.matrix_rank(Aq)

        if np.linalg.det(Aq) != 0.0:
            print(f'params get updated')
            self.adv_wt = np.linalg.solve(Aq, bq)

            if self.constrained_updates:
                self.actor.actor_wt = self.actor.param_update(
                    self.actor_lr / self.batch_size,
                    np.matmul(G, self.adv_wt),
                    self.actor.actor_wt)
            else:
                # self.actor.actor_wt -= (self.actor_lr / self.batch_size) * np.matmul(G, self.adv_wt)
                gradient_J_theta = np.matmul(G, self.adv_wt)
                new_theta = self.actor.actor_wt.cat.full() - (self.actor_lr / self.batch_size) * gradient_J_theta
                self.actor.actor_wt = self.actor.actor_wt_structure(new_theta)
                print(f'delta_theta = {np.squeeze((self.actor_lr / self.batch_size) * gradient_J_theta)}')

            print(f'self.actor.actor_wt = {self.actor.actor_wt.cat}')

    def _parse_agent_params(self, cost_params, eps, gamma, actor_lr, debug,
                            train_params, constrained_updates, experience_replay):
        self.cost_params = cost_params
        self.eps = eps
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.constrained_updates = constrained_updates
        self.experience_replay = experience_replay
        self.debug = debug
        self.iterations = train_params["iterations"]
        self.batch_size = train_params["batch_size"]


### test
if __name__ == "__main__":
    import numpy as np
    from casadi.tools import *
    import os
    import json
    from replay_buffer import BasicBuffer

    json_path = os.path.abspath(os.path.join(os.getcwd(), '../../Settings/other/house_4pumps_rl_mpc_lstd.json'))
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
    c.get_action(env.reset())
