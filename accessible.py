import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
from lib_new_parse import FindReciprocalVectors, NewMakeGrid
from math import sqrt
import sys

pi = np.pi


def R2(theta):
    s, c = np.sin(theta), np.cos(theta)
    R = np.array(
        [[c, -s], [s, c]]
    )
    return R


R = R2(0)

which = sys.argv[1]
l1, l2 = int(sys.argv[2]), int(sys.argv[3])


a1, a2 = np.array([1 / 2, sqrt(3) / 2]), np.array([-1 / 2, sqrt(3) / 2])
a3 = 2 * a2 - a1
a4 = 3 * a1 + 2 * a2
a5 = a1 - a2
a6 = a1 + a2
b1, b2 = FindReciprocalVectors(a1, a2)
# B1, B2 = FindReciprocalVectors(a1-a2, a2)
if which == "rh1":
    B1, B2 = FindReciprocalVectors(a1, a2)
if which == "rh2":
    B1, B2 = FindReciprocalVectors(a1 - a2, a1)
elif which == "re2":
    B1, B2 = FindReciprocalVectors(a1 - a2, a1 + a2)
elif which == "24":
    B1, B2 = FindReciprocalVectors(2 * a1 - a2, a1 + a2)
# B1, B2 = FindReciprocalVectors(R.dxot(a5),R.dot(a6))
KX, KY, gggg = NewMakeGrid(B1, B2, l1, l2, 3)

bz2 = ptch.RegularPolygon(
    (0, 0), 6, np.linalg.norm(
        (2 * b1 + b2) / 3), pi / 6, fill=False)
bz3 = ptch.RegularPolygon((0, 0), 6, np.linalg.norm(b1 + b2), 0, fill=False)

gggg.axes[0].add_patch(bz2)
gggg.axes[0].add_patch(bz3)
gggg.axes[0].set_xlim(-6.5, 6.5)
gggg.axes[0].set_ylim(-7.5, 7.5)
# gggg.axes[0].plot([0, b1[0]], [0, b1[1]], color='black', linestyle='-', linewidth=2)
# gggg.axes[0].plot([0, b2[0]], [0, b2[1]], color='black', linestyle='-', linewidth=2)


origin = [0], [0]
M = np.array([1 / 2, 0]).dot([b1, b2])
M23 = np.array([1 / 3, 0]).dot([b1, b2])
M43 = np.array([2 / 3, 0]).dot([b1, b2])
K12 = np.array([1 / 6, -1 / 6]).dot([b1, b2])
Kp12 = np.array([-1 / 6, 1 / 6]).dot([b1, b2])
L1 = np.array([7 / 12, -1 / 12]).dot([b1, b2])
L2 = np.array([5 / 12, 1 / 12]).dot([b1, b2])
# gross1 = np.array([3/8,7/16]).dot([b1,b2])
# gross2 = np.array([3/8,-1/16]).dot([b1,b2])
wowdude = np.array([M, M23, M43, K12, Kp12, L1, L2])
gggg.axes[0].quiver(*origin, wowdude[:, 0], wowdude[:, 1], color="r",
                    angles='xy', scale_units='xy', scale=1)
gggg.show()
