import numpy
import matplotlib.pyplot as plt
import random
import math
from scipy.optimize import minimize
from sklearn.datasets import make_circles as circle

# generate dataset
# numpy.random.seed(100)
spread = 1
classA = numpy.concatenate((numpy.random.randn(30, 2)*spread + numpy.array([0, -2]),
                            numpy.random.randn(30, 2)*spread + numpy.array([-1, 2])))
classB = numpy.concatenate((numpy.random.randn(30, 2)*spread + numpy.array([-2, -1]),
                            numpy.random.randn(30, 2)*spread + numpy.array([1.5, 1])))
inputs = numpy.concatenate((classA, classB))
targets = numpy.concatenate((numpy.ones(classA.shape[0]), -numpy.ones(classB.shape[0])))
N = inputs.shape[0]


permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]

# plot
plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
plt.axis('equal')  # force same scale on both axes
plt.savefig('svmplot.pdf')


# kernel function
#linear kernel
def k_l(xi, xj):
    return numpy.dot(xi, xj)
# polynomial kernel
def k_p(xi, xj):
    p = 3
    gamma = 1
    return (gamma*numpy.dot(xi, xj)+1)**p
# Radial basis function(RBF) kernel
def k_r(xi, xj):
    sigma = 1
    absolute = numpy.linalg.norm(xi-xj) # length of vector = (numpy.sum((xi-xj)* (xi-xj)))**0.5
    temp = (-1)*((absolute**2)/(2*(sigma**2)))
    return math.exp(temp)


# ti*tj*kernel, targets * kernel matrix
K = k_p
P = numpy.array([[targets[i] * targets[j] * K(inputs[i], inputs[j]) for j in range(N)] for i in range(N)])
def obj(alpha):
    temp = numpy.sum([numpy.dot(alpha[i]*alpha, P[:, i]) for i in range(len(alpha))])
    return 0.5 * temp - numpy.sum(alpha)
# def obj2(alpha):
#     ans = 0
#     for m in range(len(alpha)):
#         for l in range(len(alpha)):
#             ans = ans + alpha[l]*alpha[m]*P[m, l]
#     return 0.5 * ans - numpy.sum(alpha)
def zerofun(alpha):
    return numpy.dot(alpha, targets)
def indicator(new_point, nonzero, bias):
    # bias b (threshold)
    return numpy.sum([nonzero[i][2]*nonzero[i][1]*K(new_point, nonzero[i][0]) for i in range(len(nonzero))]) - bias

start = numpy.zeros(N)  # initial guess
C = 10  # slack variable (soft margin)
B = [(0, C) for _ in range(N)]  # bounds
# use function 'minimize' as a quadratic programming problem solver
ret = minimize(obj, start, bounds=B, constraints={'type': 'eq', 'fun': zerofun})
alpha = ret['x']  # print(ret['success'])

# support vectors
non_zero = []
for a in range(len(alpha)):
    if alpha[a] >= 0.00001:
        non_zero.append([inputs[a], targets[a], alpha[a]])
print(len(non_zero))

plt.plot([x[0][0] for x in non_zero], [x[0][1] for x in non_zero], 'g+')
new = numpy.random.randn(3, 2) * 1.5
print(new)
plt.plot(new[0][0], new[0][1], 'm*')
plt.plot(new[1][0], new[1][1], 'g*')
plt.plot(new[2][0], new[2][1], 'c*')
# which support vector should I use?
num_sv = 0  # choose one of the support vector, or you can also get all the results from every support vectors and then take average
b = numpy.sum([non_zero[i][2]*non_zero[i][1]*K(non_zero[num_sv][0], non_zero[i][0]) for i in range(len(non_zero))]) - non_zero[num_sv][1]
# the formula of bias is free from the effect of slack variables(ksi)(free support vectors)
t_new1 = indicator(new[0], non_zero, b)
t_new2 = indicator(new[1], non_zero, b)
t_new3 = indicator(new[1], non_zero, b)
print(t_new1, t_new2, t_new3)

# plot the decision boundary
xgrid = numpy.linspace(-6, 6)
ygrid = numpy.linspace(-4, 4)
grid = numpy.array([[indicator(numpy.array([x, y]), non_zero, b) for x in xgrid] for y in ygrid])
plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
plt.show()


