import numpy as np
from .controller import Controller
from models.manipulator_model import ManipulatorModel


class MMAController(Controller):
    def __init__(self, Tp):
        # Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3
        self.models = [ManipulatorModel(Tp, m3 = 0.1, r3 = 0.05), ManipulatorModel(Tp, m3 = 0.01, r3 = 0.01), ManipulatorModel(Tp, m3 = 1.0, r3 = 0.3)]
        self.u = np.zeros((2, 1))
        self.i = 0

    def choose_model(self, x):
        # Implement procedure of choosing the best fitting model from self.models (by setting self.i)
        q = x[:2]
        q_dot = x[2:]
        errors = np.array([])
        for model in self.models:
            model_prediction = model.M(x) @ self.u + model.C(x) @ q_dot[:, np.newaxis]
            errors = np.append(errors, np.sum(np.abs(q_dot - model_prediction)))
        self.i = np.argmin(errors)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]
        kd=25
        kp=60
        #v = q_r_ddot # add feedback
        v = q_r_ddot + kd * (q_r_dot - q_dot) + kp * (q_r - q)
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]
        return u
