import matplotlib.pyplot as plt
import numpy as np


x = [i+1 for i in range(10)]
loss_yes = [0.00245, 0.00521, 0.00364, 0.00887, 0.00030, 0.00099, 0.00122, 0.00069, 0.00008, 0.00103]
acc_yes = [1.00000, 0.98438, 0.98438, 0.96875, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000]
loss_no = [0.01344, 0.00065, 0.00521, 0.00074, 0.00020, 0.00130, 0.00040, 0.00010, 0.00634, 0.00557]
acc_no = [0.95312, 1.00000, 0.98438, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 0.95312, 0.98438]
loss_yes = [x * 100 for x in loss_yes]
loss_no = [x * 100 for x in loss_no]

fontsize = 15

# plot loss
plt.plot(x, loss_yes, linewidth=3, color="orange")
plt.plot(x, loss_no, linewidth=3)
plt.title("The change of training loss in ten epochs.", fontsize=fontsize)
plt.xlabel("Number of epochs", fontsize=fontsize)
plt.ylabel(r'Trainig loss ($\times 10^{-2}$)', fontsize=fontsize)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(["With BatchNorm","Without BatchNorm"], loc="upper right", fontsize=fontsize)
plt.grid(True)
plt.savefig('loss_all.png')
plt.show()


# plot acc
plt.plot(x, acc_yes, linewidth=3, color="orange")
plt.plot(x, acc_no, linewidth=3)
plt.title("The change of training accuracy in ten epochs.", fontsize=fontsize)
plt.xlabel("Number of epochs", fontsize=fontsize)
plt.ylabel("Trainig accuracy", fontsize=fontsize)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(["With BatchNorm","Without BatchNorm"], loc="best", fontsize=fontsize)
plt.grid(True)
plt.savefig('acc_all.png')
plt.show()

