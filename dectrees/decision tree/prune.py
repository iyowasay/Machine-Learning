import monkdata as m
import dtree as dt
import random
import matplotlib.pyplot as plt
import numpy as np
print('Pruning')
# random partition into training and validation
def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]


monk1train, monk1val = partition(m.monk1, 0.6)
tree4 = dt.buildTree(monk1train, m.attributes)
tree5 = dt.buildTree(monk1train, m.attributes)
cur_res = dt.check(tree4, monk1val)
print(cur_res, tree4)

fraction = [0.3,0.4,0.5,0.6,0.7,0.8]
cycle = 1000
res1, res3 = [], []
for f in fraction:
    c1, c3 = 0, 0
    for _ in range(cycle):
        monk1train, monk1val = partition(m.monk1, f)
        monk3train, monk3val = partition(m.monk3, f)
        tree1 = dt.buildTree(monk1train, m.attributes)
        tree3 = dt.buildTree(monk3train, m.attributes)
        al1 = dt.allPruned(tree1)
        al3 = dt.allPruned(tree3)
        cur1 = dt.check(tree1, monk1val)
        cur3 = dt.check(tree3, monk3val)
        for j in al1:
            if dt.check(j, monk1val) >= cur1:
                tree1 = j
                cur1 = dt.check(j, monk1val)
        for k in al3:
            if dt.check(k, monk3val) >= cur3:
                tree3 = k
                cur3 = dt.check(k, monk3val)
        c1 += dt.check(tree1, m.monk1test)
        c3 += dt.check(tree3, m.monk3test)
    res1.append(c1/cycle)
    res3.append(c3/cycle)

print(res1, res3)
fig, ax = plt.subplots()
index = np.arange(len(fraction))
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index - bar_width/2, res1, bar_width, alpha=opacity, color='b', label='Monk1')
rects2 = plt.bar(index + bar_width/2, res3, bar_width, alpha=opacity, color='g', label='Monk3')

plt.xlabel('Fraction')
plt.ylabel('Correctness')
plt.title('Performance using different fraction (Mean value of 1000 cycle)')
plt.xticks(index, ('0.3', '0.4', '0.5', '0.6', '0.7', '0.8'))
plt.legend()

plt.tight_layout()
plt.show()


# def prune2(tar, vali, li):
#     best_per = dt.check(tar, vali)
#     count = 0
#     al = dt.allPruned(tar)
#     for i in al:
#         check = dt.check(i, vali)
#         if check >= best_per:
#             count += 1
#             prune2(i, vali, li)
#             best_per = check
#     if count == 0:
#         li.append((best_per, tar))
#     print(count)







