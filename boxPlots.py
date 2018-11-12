
import numpy as np
from matplotlib.pyplot import boxplot, figure, show, xticks, ylabel

ann = np.array([153.05555392,  79.18352584,  76.15750333,  36.84525125, 266.52780268])
np.mean(ann)
LinReg = np.array([200.1924, 136.7886, 127.6576, 87.61300, 76.13174])
average_error = np.array([312.415052946, 240.42194349, 232.775148788, 220.761348449, 152.638906755])

data = [ann, LinReg, average_error]

figure()
boxplot(data)
ylabel("Generalization Error")
xticks(np.arange(4), (" ", "ANN", "Linear Regression", "Average Output"))
show()

