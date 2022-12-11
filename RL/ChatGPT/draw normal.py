import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
mean = 0.050004
std = 0.2179535

x = np.linspace(-5,5,100)

y = norm.pdf(x,mean,std)
plt.plot(x,y)
plt.show()
