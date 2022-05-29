from cProfile import label
from turtle import color
import matplotlib.pyplot as plt

from matplotlib.pyplot import MultipleLocator
import numpy as np



transcost_list = [3.917,3.005,2.623,2.329,2.074,1.965,1.921,1.854,1.787,1.762,1.718,1.652,1.785,1.575,1.560,1.540,1.563,1.476]

cnncost_list =[4.506,3.362,2.976,2.510,2.394,2.347,2.216,2.004,2.004,1.927,1.922,1.888,1.856,1.826,1.797,1.748,1.724,1.682]

epoch = np.arange(1,19)



x_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.title("2D convolution Correlation Volume Vs Transformer Correlation Cost Volume ")

plt.plot(epoch,transcost_list,'o',color='blue')

plt.plot(epoch,transcost_list,label='Trans',color='blue')
plt.plot(epoch,cnncost_list,'*',color='green')
plt.plot(epoch,cnncost_list,label='CNN',color='green')
plt.xlabel("Epoch")
plt.ylabel("EPE")

plt.legend()
plt.show()