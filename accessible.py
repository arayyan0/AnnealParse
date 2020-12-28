import matplotlib.pyplot as plt
import matplotlib.patches as ptch
from common import a1, a2, b1, b2, FindReciprocalVectors, KMeshForPlotting, RotateIn2D
import sys

R = RotateIn2D(0)

which = sys.argv[1]
l1, l2 = int(sys.argv[2]), int(sys.argv[3])

if which == "rh1":
    B1, B2 = FindReciprocalVectors(a1, a2)
elif which == "rh2":
    B1, B2 = FindReciprocalVectors(a1 - a2, a1)
elif which == "re1":
    B1, B2 = FindReciprocalVectors(a1, 2*a2-a1)
elif which == "re2":
    B1, B2 = FindReciprocalVectors(a1-a2, a1+a2)
elif which == "24":
    B1, B2 = FindReciprocalVectors(2*a1-a2, a1+a2)

_, _, fig = KMeshForPlotting(B1, B2, l1, l2, 2, 2, True, True)

plt.show()
plt.close()
