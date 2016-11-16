import numpy as np
import matplotlib.pyplot as plt

class PDF(object):

    def __init__(self, truth=None):
        self.truth = truth

    def evaluate(self, loc):#PDF

        return

    def integrate(self, limits):#CDF

        return

    def quantize(self, num_points):#inverse CDF (PPF)

        quanta = 1./num_points
        points = np.linspace(0.+quanta, 1.-quanta, num_points)
        print(points)
        self.quantiles = self.truth.ppf(points)
        print(self.quantiles)

        return self.quantiles

    def interpolate(self):

        return

    def plot(self, limits, num_points):

        x = np.linspace(limits[0], limits[1], 100)

        y = [0., 1.]
        plt.plot(x, self.truth.pdf(x), color='k', linestyle='-', lw=1., alpha=1., label='true pdf')
        plt.vlines(self.quantize(num_points), y[0], y[1], color='k', linestyle='--', lw=1., alpha=1., label='quantiles')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('probability density')
        plt.savefig('plot.png')

        return
