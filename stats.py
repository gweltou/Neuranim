#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt


class Stats:
    def __init__(self):
        self.var_dict = {}
    
    def feed(self, *values):
        if len(self.var_dict) == 0:
            for var, v in enumerate(values):
                self.var_dict[var] = [v]
        else:
            for var, v in enumerate(values):
                self.var_dict[var].append(v)
        #print(self.var_dict)
    
    def savePlot(self, filename, title=''):
        plt.close()
        plt.plot(self.var_dict[0], self.var_dict[1], label='gen score')
        plt.plot(self.var_dict[0], self.var_dict[2], label='best')
        plt.plot(self.var_dict[0], self.var_dict[3], label='worst')
        plt.legend()
        plt.title(title)
        plt.savefig(filename)

    def reset(self):
        self.var_dict = {}
