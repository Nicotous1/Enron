import matplotlib.pyplot as plt
from SVBM import *

net = HofWig_SBM(100,5)
s = SVBM(net)

s.run(2,8)

s.plot_lower_bound()
s.plot_matrix(5)

plt.show()
