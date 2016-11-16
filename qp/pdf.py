class PDF(object):

    def __init__(self, truth=None):
        self.truth = truth

    def evaluate(self, loc):

        return

    def integrate(self, bounds):

        return

    def quantize(self):

        return

    def interpolate(self):

        return

    def plot(self, limits):

        x = np.linspace(limits[0], limits[1], 100)
        plt.plot(x, self.truth.pdf(x), 'r-', lw=5, alpha=0.6, label='true pdf')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('probability density')
        plt.savefig('plot.png')

        return