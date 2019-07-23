import numpy as np
import math

class Prediction:
    pos_x_og = 0
    pos_y_og = 0
    vel_x_og = 1
    vel_y_og = 1
    noise_value = 0.4

    real_pos_noise_arr = []
    real_vel_noise_arr = []

    def __init__(self, sig_v, sig_p, pos_x, pos_y, width):
        self.pos_x_og = pos_x
        self.pos_y_og = pos_y
        self.width = width

        self.arr_f = np.array([
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ])
        self.arr_q = [
            [self.noise_value, 0, 0, 0, 0],
            [0, self.noise_value, 0, 0, 0],
            [0, 0, self.noise_value, 0, 0],
            [0, 0, 0, self.noise_value, 0],
            [0, 0, 0, 0, self.noise_value]
        ]
        self.arr_r = np.array([
            [self.noise_value, 0, 0],
            [0, self.noise_value, 0],
            [0, 0, self.noise_value],
        ])
        #Check
        self.arr_h = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1]
        ])

        self.p_k = self.arr_q
        self.x_k = np.array([
            pos_x,
            pos_y,
            1,
            1,
            width
        ])

    def noise_w(self, sigma):
        return np.random.normal(0, sigma)

    def predict(self):
        noise_pos_x = self.noise_w(00.4)
        noise_pos_y = self.noise_w(00.4)
        noise_vel_x = self.noise_w(00.4)
        noise_vel_y = self.noise_w(00.4)
        noise_width = self.noise_w(00.4)

        w = np.array([noise_pos_x, noise_pos_y, noise_vel_x, noise_vel_y, noise_width])

        print("x_k antiguo")
        print(self.x_k)
        self.x_k = np.dot(self.arr_f, self.x_k) + w
        print("x_k predecido ------------------------------")
        print(self.x_k)    

    def update(self, pos_x, pos_y, width):
        print("x_k antiguo")
        print(self.x_k)
        new_pos = np.array([
            pos_x,
            pos_y,
            width
        ])

        self.p_k = np.dot(self.arr_f, np.dot(
                self.p_k, self.arr_f.transpose())) + self.arr_q
        print(self.p_k.shape)
        
         # Filtrado
        temp = np.linalg.inv(
            np.add(np.dot(np.dot(self.arr_h, self.p_k), self.arr_h.transpose()), self.arr_r))
        k_k = np.dot(np.dot(self.p_k, self.arr_h.transpose()), temp)

        _temp = np.array([
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ])


        __temp = (new_pos - np.dot(_temp, self.x_k))
        __temp = np.dot(k_k, __temp) 
        self.x_k = self.x_k + __temp

        # y_k = arr_z_k[i] - self.arr_h.dot(x_k)
        # temp = np.dot(k_k, y_k)
        # x_k = np.add(x_k, temp)
        # filter_positions.append(x_k)
        # nx = len(self.arr_q)
        # Inx = np.eye(nx)
        # p_k = np.dot((np.subtract(Inx, np.dot(k_k, self.arr_h))), p_k)

        print("x_k nuevo updateado *****************")
        print(self.x_k)

    def get_positions(self):

        height = self.x_k[4] * 1.35
        pos = [
            int(self.x_k[0]),
            int(self.x_k[1]),
            int(self.x_k[0] + self.x_k[4]),
            int(self.x_k[1] + height)
        ]
        return pos


    def get_vector(self, array, pos):
        vector = []
        for i in array:
            vector.append(i[pos])
        return vector
