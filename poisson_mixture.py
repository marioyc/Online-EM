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
                u -= self.w[i]

def poisson_mixture_online_em(n, gamma_0, alpha, sampler, averaging=False, n0=None):
    assert gamma_0 > 0 and gamma_0 < 1 and alpha > 0.5 and alpha <= 1

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

    if averaging and n0 is None:
        n0 = n / 2

    for i in range(n):
        y = sampler.sample()
        gamma = gamma_0 / (i + 1) ** alpha

        for j in range(m):
            w_i[j] = w[j] * param[j]**y * numpy.exp(-param[j])
        w_i /= numpy.sum(w_i)

        s_temp[:, 0] = w_i
        s_temp[:, 1] = w_i * y
        s = s + gamma * (s_temp - s)

        w = s[:, 0]
        param = s[:, 1] / s[:, 0]

        if averaging and i > n0:
            w_seq[i, :] = (w_seq[i - 1, :] * (i - n0 + 1) + w) / (i - n0 + 2)
            param_seq[i, :] = (param_seq[i - 1, :] * (i - n0 + 1) + param) / (i - n0 + 2)
        else:
            w_seq[i, :] = w
            param_seq[i, :] = param

    print(w_seq[-1, :], param_seq[-1, :])
    return w_seq, param_seq

w = [1.0 / 6, 1.0 / 3, 1.0 / 2]
param = [2.5, 5, 10]
#w = [1.0 / 3, 2.0 / 3]
#param = [2, 3]
sampler = PoissonMixtureSampler(w, param)

w_seq, param_seq = poisson_mixture_online_em(10000, 0.5, 1, sampler)
w_seq2, param_seq2 = poisson_mixture_online_em(10000, 0.5, 0.6, sampler)
w_seq3, param_seq3 = poisson_mixture_online_em(10000, 0.5, 0.6, sampler, averaging=True)

f, axarr = plt.subplots(sampler.m, 2, sharex=True)

for i in range(sampler.m):
    axarr[i, 0].set_title(r'$w_{}$ = {:.5f}'.format(i + 1, w[i]))
    axarr[i, 0].plot(w_seq[:, i], label=r'$\alpha = 1$')
    axarr[i, 0].plot(w_seq2[:, i], label=r'$\alpha = 0.6$')
    axarr[i, 0].plot(w_seq3[:, i], label=r'$\alpha = 0.6$, averaging')
    legend1 = axarr[i, 0].legend()

    axarr[i, 1].set_title(r'$\lambda_{}$ = {:.5f}'.format(i + 1, param[i]))
    axarr[i, 1].plot(param_seq[:, i], label=r'$\alpha = 1$')
    axarr[i, 1].plot(param_seq2[:, i], label=r'$\alpha = 0.6$')
    axarr[i, 1].plot(param_seq3[:, i], label=r'$\alpha = 0.6$, averaging')
    legend2 = axarr[i, 1].legend()

plt.show()
