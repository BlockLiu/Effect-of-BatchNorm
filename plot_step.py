import matplotlib.pyplot as plt
import numpy as np

loss_yes, loss_no, acc_yes, acc_no = [], [], [], []

with open('log_with_bn.txt', 'r') as fin:
	for line in fin.readlines():
		l = line.split(' ')
		loss_yes.append(float(l[0]))
		acc_yes.append(float(l[1][:-1]))		

with open('log_without_bn.txt', 'r') as fin:
	for line in fin.readlines():
		l = line.split(' ')
		loss_no.append(float(l[0]))
		acc_no.append(float(l[1][:-1]))
		
x = [i+1 for i in range(len(loss_yes))]		

fontsize = 15

# plot loss
plt.plot(x, loss_yes, linewidth=1, color="orange")
plt.plot(x, loss_no, linewidth=1)
plt.title("The change of training loss in the first epoch.", fontsize=fontsize)
plt.xlabel("Number of steps", fontsize=fontsize)
plt.ylabel("Trainig loss", fontsize=fontsize)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(["With BatchNorm","Without BatchNorm"], loc="upper right", fontsize=fontsize)
plt.grid(True)
plt.savefig('loss.png')
plt.show()


# plot acc
plt.plot(x, acc_yes, linewidth=1, color="orange")
plt.plot(x, acc_no, linewidth=1)
plt.title("The change of training accuracy in the first epoch.", fontsize=fontsize)
plt.xlabel("Number of steps", fontsize=fontsize)
plt.ylabel("Trainig accuracy", fontsize=fontsize)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(["With BatchNorm","Without BatchNorm"], loc="lower right", fontsize=fontsize)
plt.grid(True)
plt.savefig('acc.png')
plt.show()



