import operator as op
import numpy as np

x = np.array([0.644, 122.546, -9.129,
              0.675, 0.36, 0.0372, 0.533,122.358, -9.8,0.323, 0.722, 0.0297])
count = {}
for i in x[0:len(x) - 1]:
    count[i] = count.get(i, 0) + 1
count = sorted(count.items(), key=op.itemgetter(0), reverse=False)

markov_marix = np.zeros([len(count), len(count)])
for j in range(len(x) - 1):
    for m in range(len(count)):
        for n in range(len(count)):
            if x[j] == count[m][0] and x[j + 1] == count[n][0]:
                markov_marix[m][n] += 1
for t in range(len(count)):
    markov_marix[t, :] /= count[t][1]
print(markov_marix)
a, b = np.linalg.eig(markov_marix)
print(b)
