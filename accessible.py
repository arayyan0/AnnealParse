import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import numpy as np
from common import a1, a2, b1, b2, FindReciprocalVectors, KMeshForPlotting, RotateIn2D, SymmetryPoints, pi
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
elif which == "c3":
    B1, B2 = FindReciprocalVectors(2*a1-a2, a1+a2)

_, _, fig = KMeshForPlotting(B1, B2, l1, l2, 2, 2, True, True)

_, _, sym_labels, sym_points = SymmetryPoints().MakeKPath(["X","G","M2","Gp1","M1","G"],50)
sym_points = np.array(sym_points)/2/pi

ax = fig.axes[0]
ax.scatter(sym_points[:,0], sym_points[:,1],c='white')
fig.axes[0].set_facecolor('black')

ax.annotate(r'$\;\;$'+sym_labels[0], (sym_points[0,0], sym_points[0,1]),c='white',fontsize=12)
ax.annotate(r'$\;\;$'+sym_labels[1], (sym_points[1,0], sym_points[1,1]),c='white',fontsize=12)
ax.annotate(r'$\;\;$'+sym_labels[2], (sym_points[2,0], sym_points[2,1]),c='white',fontsize=12)
ax.annotate(r'$\;\;$'+sym_labels[3], (sym_points[3,0], sym_points[3,1]),
                                    xytext =  (sym_points[3,0]-0.2, sym_points[3,1]-0.1),
                                    c='white',fontsize=12)
ax.annotate(r'$\;\;$'+sym_labels[4], (sym_points[4,0], sym_points[4,1]),c='white',fontsize=12)

plt.show()
# plt.savefig('accessible.pdf')
plt.close()
