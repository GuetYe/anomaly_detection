from matplotlib import pyplot as plt
import numpy as np
import random
basevalue = [(i+1) * 2 for i in range(10)]
print(basevalue)
res = np.zeros([10,25])
for i in range(10):
    for xp in range(25):
        inc_value = random.uniform(-0.5,0.5)
        res[i][xp] = basevalue[i] + inc_value
print(res)
for i in range(10):
    plt.plot([sd for sd in range(25)],res[i])
plt.show()