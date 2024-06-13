import numpy as np
from models.manipulator_model import ManipulatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManipulatorModel(Tp, m3=2)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """
        q=x[:2]
        q_dot=x[2:]
        kd=20
        kp=60
        v = q_r_ddot + kd * (q_r_dot - q_dot) + kp * (q_r - q)
        tau = self.model.M(x) @ v + self.model.C(x) @ q_dot
        return tau
