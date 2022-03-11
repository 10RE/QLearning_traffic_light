import numpy as np
import random as r
import matplotlib.pyplot as plt
import json
import os


class Intersection:
    def __init__(self, capacity):
        self.cars = [[], [], [], []]  # 上右下左，包含每个车的等待时间
        self.spawn_rate = 0.3
        self.capacity = capacity
        self.light_status = 0  # 0左右开， 1上下开
        self.count_down = 0  # 计量黄灯时间
        # self.nextcars = [[0, 0, 0, 0]

    def reset(self):
        self.cars = [[], [], [], []]
        self.light_status = 0
        self.count_down = 0

    def is_end_state(self):
        return sum([len(self.cars[i]) >= self.capacity for i in range(4)])

    def get_score(self):
        return self.cal_score(self.cars, self.countdown)

    def cal_score(self, carstate, countdownnum):
        score = 1000
        score -= sum([sum(x) for x in carstate])
        score -= countdownnum * 2
        return score

    def cal_reward(self, action):
        reward = self.capacity
        if ((action == 0) == (self.light_status == 0)):
            # for i in [0, 2]:
            # reward += int(len(self.cars[i]) * 0.3)
            reward += max([int(len(self.cars[i]) * 0.3) for i in [0, 2]])
            # for i in [1, 3]:
            #    reward -= ((len(self.cars[i]) * len(self.cars[i])) / self.capacity)
            reward -= max([((len(self.cars[i]) * len(self.cars[i])) / self.capacity) for i in [1, 3]])
        else:
            # for i in [1, 3]:
            #    reward += int(len(self.cars[i]) * 0.3)
            reward += max([int(len(self.cars[i]) * 0.3) for i in [1, 3]])
            # for i in [0, 2]:
            #    reward -= ((len(self.cars[i]) * len(self.cars[i])) / self.capacity)
            reward -= max([((len(self.cars[i]) * len(self.cars[i])) / self.capacity) for i in [0, 2]])

        # if ((action == 1) and len(self.cars[i]))

        # if ((action == 0) == (self.light_status == 0)):
        # for i in [0, 2]:
        #    if (len(self.cars[i]) > self.capacity - 3):
        #        reward +=
        #    reward += ((len(self.cars[i]) ** 2) / self.capacity)
        # for i in [1, 3]:
        # reward -= ((len(self.cars[i]) * len(self.cars[i])) / self.capacity)
        # if len(self.cars[i]) > 0:
        #    reward += self.cars[i][0]
        # reward += 1
        # else:
        # for i in [0, 2]:
        # reward -= ((len(self.cars[i]) * len(self.cars[i])) / self.capacity)
        # for i in [1, 3]:
        #    reward += ((len(self.cars[i]) ** 2) / self.capacity)
        # if len(self.cars[i]) > 0:
        #    reward += self.cars[i][0]
        # reward += 1

        # reward -= self.count_down * 2
        # if (self.count_down != 0 and action == 1):
        #    reward -= 999999
        # for lane in self.cars:
        #    reward -= (len(lane) ** 2 / self.capacity)
        return reward

    def next_state(self, action):
        action = int(action)
        local_count_down = self.count_down
        if (local_count_down != 0):
            local_count_down -= 1
        local_nextcars = self.cars.copy()
        for i in range(4):
            if (r.random() < self.spawn_rate):
                local_nextcars[i].append(0)
        # print("action: {}".format(action))
        if (action == 0):
            # print(local_count_down)
            if (local_count_down == 0):
                # print(self.light_status)
                if (self.light_status == 0):
                    for i in [0, 2]:
                        if len(local_nextcars[i]) > 0:
                            local_nextcars[i].pop(0)
                else:
                    for i in [1, 3]:
                        if len(local_nextcars[i]) > 0:
                            local_nextcars[i].pop(0)
        else:
            # self.light_status = not self.light_status
            local_count_down = 2
        for i in range(len(local_nextcars)):
            for j in range(len(local_nextcars[i])):
                local_nextcars[i][j] += 1

        return (local_nextcars, local_count_down)

    def take(self, action, demo=False):
        action = int(action)
        if (demo):
            if (self.count_down != 0 and action == 1):
                print('action not taken')
                action = 0
        self.cars, self.count_down = self.next_state(action)
        if (action != 0):
            self.light_status = not self.light_status
        return

    def get_state(self):
        return [len(x) for x in self.cars]

    def cal_state(self, state):
        return [len(x) for x in state]

    def print_state(self):
        print(self.cars)
        print('light: {}, count: {}'.format(self.light_status, self.count_down))


def l2s(l_list):
    t_list = [str(num) for num in l_list]
    return ' '.join(t_list)


class QLP:
    def __init__(self, alpha, epsilon, gamma, num_train, capacity):
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.num_train = int(num_train)
        self.capacity = capacity
        self.qtable = {}

        curKey = [0, 0, 0, 0]
        for i in range(capacity ** 4):
            for j in range(4):
                if (curKey[3 - j] == capacity - 1):
                    curKey[3 - j] = 0
                else:
                    curKey[3 - j] += 1
                    break
            for i in range(3):
                for j in range(2):
                    self.qtable[str(l2s(curKey + [i] + [j]))] = {'0': 0, '1': 0}
        # print(self.qtable)

    def chose_action(self, state):
        if r.random() > self.epsilon:
            return max(self.qtable[l2s(state)], key=self.qtable[l2s(state)].get)
        else:
            return 1 if r.random() < 0.5 else 0

    def train(self):
        cross = Intersection(self.capacity)
        train_step = 0
        while train_step < self.num_train:
            cross.reset()
            while not cross.is_end_state():
                if (train_step > self.num_train):
                    break
                print(train_step)
                state = cross.get_state() + [cross.count_down] + [int(cross.light_status)]
                action = self.chose_action(state)
                next_state, next_countdown = cross.next_state(action)
                if (sum([len(next_state[i]) >= self.capacity for i in range(4)]) != 0):
                    break
                reward = cross.cal_score(next_state, next_countdown)
                reward = cross.cal_reward(action)
                next_state = cross.cal_state(next_state) + [next_countdown] + [
                    int(cross.light_status) if action == 0 else int(not cross.light_status)]

                sp_score = max(self.qtable[l2s(next_state)]['0'], self.qtable[l2s(next_state)]['1'])
                cur_q = (1 - self.alpha) * self.qtable[l2s(state)][str(action)] + self.alpha * (
                            reward + self.gamma * sp_score)
                self.qtable[l2s(state)][str(action)] = cur_q
                cross.take(action)
                train_step += 1

    def get_action(self, state, print_=True):
        if (print_):
            print('q val 0 = {}, q val 1 = {}, optimal = {}'.format(self.qtable[l2s(state)]['0'],
                                                                    self.qtable[l2s(state)]['1'],
                                                                    max(self.qtable[l2s(state)],
                                                                        key=self.qtable[l2s(state)].get)))
        return max(self.qtable[l2s(state)], key=self.qtable[l2s(state)].get)

    def dump(self, file_name):
        with open(file_name, 'w') as fp:
            json.dump(self.qtable, fp)

    def load(self, file_name):
        with open(file_name, 'r') as fp:
            self.qtable = json.load(fp)


def demo(qlp, demo_range=10, print_=True):
    cross = Intersection(qlp.capacity)
    cost = 0
    for i in range(demo_range):
        if (print_):
            print('\n# {}'.format(i))
        state = cross.cars
        for lane in state:
            cost += len(lane) * len(lane)
        if (print_):
            cross.print_state()
        state = cross.cal_state(state) + [cross.count_down] + [int(cross.light_status)]
        try:
            action = qlp.get_action(state, print_)
        except KeyError:
            break
            # if (cross.light_status == 1 and max([len(cross.cars[i]) for i in [0, 2]]) > 15):
            #    action = 1
            # elif (cross.light_status == 0 and max([len(cross.cars[i]) for i in [1, 3]]) > 15):
            #    action = 1
        if (print_):
            print(action)
        cross.take(action, True)
    if (print_):
        print('avg cost = {}'.format(cost / demo_range))
    return cost / demo_range


def demo_contrast(demo_range=10, print_=True):
    cross = Intersection(15)
    cost = 0
    for i in range(demo_range):
        if (print_):
            print('\n# {}'.format(i))
        state = cross.cars
        for lane in state:
            cost += len(lane) * len(lane)
        if (print_):
            cross.print_state()
        action = (i % 6 == 0)
        if (print_):
            print(action)
        cross.take(action, True)
    if (print_):
        print('avg cost = {}'.format(cost / demo_range))
    return cost / demo_range


if (__name__ == '__main__'):
    ql = QLP(0.5, 0.4, 0.5, 1000000, 15)
    file_name = 'qtable.json'
    if (os.path.exists(file_name)):
        print('json loaded!')
        ql.load(file_name)
    else:
        ql.train()
        print('json dumped!')
        ql.dump(file_name)
    # print(ql.qtable)
    # print(ql.qtable)
    ori_data = []
    con_data = []
    for i in range(5000):
        print(i)
        ori_data.append(demo(ql, 500, False))
        con_data.append(demo_contrast(500, False))

    plt.boxplot([ori_data, con_data])
    plt.show()