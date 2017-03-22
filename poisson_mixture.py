import matplotlib.pyplot as plt
import numpy

class PoissonMixtureSampler:
    def __init__(self, w, param):
        assert len(w) == len(param)
        self.w = w
        self.param = param
        self.m = len(w)

    def sample(self):
        u = numpy.random.rand()

        for i in range(self.m):
            if u <= self.w[i]:
                return numpy.random.poisson(self.param[i])
            else:
                u -= w[i]

def poisson_mixture_online_em(n, gamma_0, sampler):
    m = sampler.m
    w_seq = numpy.zeros((n, m))
    w = numpy.ones(m) / m
    w_i = numpy.zeros(m)
    param_seq = numpy.zeros((n, m))
    param = numpy.arange(1, m + 1)
    s = numpy.zeros((m, 2))
    s[:, 0] = 1.0 / m
    s[:, 1] = numpy.arange(1, m + 1)
    s_temp = numpy.zeros((m, 2))

    for i in range(n):
        y = sampler.sample()
        gamma = gamma_0 / (i + 1)

        for j in range(m):
            w_i[j] = w[j] * param[j]**y * numpy.exp(-param[j])
        w_i /= numpy.sum(w_i)

        s_temp[:, 0] = w_i
        s_temp[:, 1] = w_i * y
        s = s + gamma * (s_temp - s)

        w = s[:, 0]
        param = s[:, 1] / s[:, 0]
        w_seq[i, :] = w
        param_seq[i, :] = param

    print(w, param)
    return w_seq, param_seq

w = [1.0 / 6, 1.0 / 3, 1.0 / 2]
param = [2.5, 5, 10]
sampler = PoissonMixtureSampler(w, param)

w_seq, param_seq = poisson_mixture_online_em(1000, 0.5, sampler)

f, axarr = plt.subplots(sampler.m, 2, sharex=True)

for i in range(sampler.m):
    axarr[i, 0].plot(w_seq[:, i])
    axarr[i, 1].plot(param_seq[:, i])

plt.show()
