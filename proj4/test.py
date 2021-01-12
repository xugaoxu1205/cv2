import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

trains = pd.read_table('train loss.txt', header=None)
train_accs = []
train_losss = []
for i in np.arange(0, trains.shape[0], step=10):
    train_acc = trains[0][i][-8:-2]
    train_loss = trains[0][i][-21:-16]
    train_accs.append(float(train_acc))
    train_losss.append(float(train_loss))
print(train_losss)
print(train_accs)

plt.plot(np.arange(0, trains.shape[0], step=10), train_losss)

plt.title("train loss of resnet")
plt.xlabel('iters')
plt.ylabel('loss')
plt.yticks(np.arange(0.2,3,0.5))
plt.show()
