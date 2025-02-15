
import matplotlib.pyplot as plt
import numpy as np
import numpy
from scipy import signal

def f(x):
    y = (9*x+6)*np.sin(x) + 96*x
    return y


def f2(x):
    return (x-30)**2 + 3*x + 1220

def mycov(y1, y2):
    assert len(y1) == len(y2), "lenght mismatch"
    prumer1 = np.mean(y1)
    prumer2 = np.mean(y2)

    cov1 = y1 - prumer1
    cov2 = y2 - prumer2

    return 1/len(y1)*sum(cov1*cov2)


x = np.linspace(0, 20*np.pi, 10000)

target = {"sinovy signal":np.sin(x), "obdelnikovy signal":signal.square(x), "pilovy signal":signal.sawtooth(x), "trojuhelnikovy signal": signal.sawtooth(x,width=0.5)}

for iden,sig in target.items():
    for tr_iden,tr in target.items():
        if iden == tr_iden:
            continue
        plt.plot(sig)
        plt.plot(tr)
        plt.title(f"kovariance {iden} a {tr_iden} je:\n {np.cov(sig, tr)}")
        plt.show()

# y1 = f(x)
# y2 = f2(x)

# plt.plot(y1, label="2x+6")
# plt.plot(y2, label="3x-12")

# print(mycov(y1, y2))

plt.show()