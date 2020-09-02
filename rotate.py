from lib_new_parse import LocalRotation
import numpy as np
import numpy.linalg as LA

z = 0.4
x = np.sqrt((1 - z**2) / 2)

moment = np.array([x, x, z])

moment2 = np.array([0.65409702417487, 0.64360136952274, 0.39740956218244])

print(LA.norm(moment))

caxis = np.sqrt(np.ones(3) / 3)


print(LocalRotation(caxis).dot(caxis))
print(LocalRotation(caxis).dot(moment))
print(LocalRotation(caxis).dot(moment2))
