import strawberryfields as sf
from strawberryfields.ops import *

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

prog = sf.Program(1)

with prog.context as q:
    Vac | q[0]

eng = sf.Engine('gaussian')
state = eng.run(prog).state

fig = plt.figure()
X = np.linspace(-7, 7, 100)
P = np.linspace(-7, 7, 100)
Z = state.wigner(0, X, P)
X, P = np.meshgrid(X, P)
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, P, Z, cmap="RdYlGn", lw=0.5, rstride=1, cstride=1)
fig.set_size_inches(4.8, 5)
ax.set_axis_off()
plt.savefig('wigner_func_vacuum_state', dpi=600)

prog = sf.Program(1)

with prog.context as q:
    Sgate(1.0) | q[0]

eng = sf.Engine('gaussian')
state = eng.run(prog).state

fig = plt.figure()
X = np.linspace(-7, 7, 100)
P = np.linspace(-7, 7, 100)
Z = state.wigner(0, X, P)
X, P = np.meshgrid(X, P)
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, P, Z, cmap="RdYlGn", lw=0.5, rstride=1, cstride=1)
fig.set_size_inches(4.8, 5)
ax.set_axis_off()

plt.savefig('wigner_func_squeezed_state', dpi=600)
plt.show()