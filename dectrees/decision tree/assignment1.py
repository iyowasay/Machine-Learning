import monkdata as m
import dtree as dt
import drawtree_qt5 as draw
import random
import matplotlib

ans1 = dt.entropy(m.monk1)
ans2 = dt.entropy(m.monk2)
ans3 = dt.entropy(m.monk3)

print('Entropy monk1', ans1)
print('Entropy monk2', ans2)
print('Entropy monk3', ans3)
print()
print('Information Gain')
print('Monk1')
for i in range(6):
    print(round(dt.averageGain(m.monk1, m.attributes[i]), 5), end = ' ')
print()
print('Monk2')
for i in range(6):
    print(round(dt.averageGain(m.monk2, m.attributes[i]), 5), end = ' ')
print()
print('Monk3')
for i in range(6):
    print(round(dt.averageGain(m.monk3, m.attributes[i]), 5), end = ' ')

print()
print()
attr1 = m.attributes[4]
attr2 = m.attributes[4]
attr3 = m.attributes[1]

print('Assignment 4')
print()
print('Monk1')
sub1 = [[] for _ in range(len(attr1.values))]
for k in range(len(attr1.values)):
    sub1[k] = dt.select(m.monk1, attr1, attr1.values[k])
    print('subset size', len(sub1[k]), ', subset entropy', dt.entropy(sub1[k]))
print('Monk2')
sub2 = [[] for _ in range(len(attr2.values))]
for k in range(len(attr2.values)):
    sub2[k] = dt.select(m.monk2, attr2, attr2.values[k])
    print('subset size', len(sub2[k]), ', subset entropy', dt.entropy(sub2[k]))
print('Monk3')
sub3 = [[] for _ in range(len(attr3.values))]
for k in range(len(attr3.values)):
    sub3[k] = dt.select(m.monk3, attr3, attr3.values[k])
    print('subset size', len(sub3[k]), ', subset entropy', dt.entropy(sub3[k]))
print()
print('5. Build Tree for monk1 - choose A5 as first split')

for k in range(4):
    temp = 0
    mx = 0
    print('Node', k+1)
    for i in range(6):
        if i != 4:
            tt = dt.averageGain(sub1[k], m.attributes[i])
            # tt = round(dt.averageGain(sub[k], m.attributes[i]), 5)
            print(tt)
            if tt > mx:
                temp = i+1
                mx = tt
    if mx == 0:
        print('This is a leaf! No need to be tested.')
    else:
        print('At Node', k+1, '- A', temp, 'should be tested')

print()
print('Most common')
print('Node1 - A5', dt.mostCommon(sub1[0]))
print('Node2 - A4')
q = m.attributes[3]
for i in range(len(q.values)):
    qq = dt.select(sub1[1], q, q.values[i])
    print('for value', q.values[i], '-', dt.mostCommon(qq))
print('Node3 - A6')
q = m.attributes[5]
for i in range(len(q.values)):
    qq = dt.select(sub1[2], q, q.values[i])
    print('for value', q.values[i], '-', dt.mostCommon(qq))
print('Node4 - A1')
q = m.attributes[0]
for i in range(len(q.values)):
    qq = dt.select(sub1[3], q, q.values[i])
    print('for value', q.values[i], '-', dt.mostCommon(qq))

# tre = dt.buildTree(m.monk1, m.attributes, 2)
# draw.drawTree(tre)
print()
print('Performance of fully grown trees')
print('Monk1 - ')
tre1 = dt.buildTree(m.monk1, m.attributes)
print(dt.check(tre1, m.monk1), dt.check(tre1, m.monk1test))
# draw.drawTree(tre1)
print('Monk2 - ')
tre2 = dt.buildTree(m.monk2, m.attributes)
print(dt.check(tre2, m.monk2), dt.check(tre2, m.monk2test))
# draw.drawTree(tre2)
print('Monk3 - ')
tre3 = dt.buildTree(m.monk3, m.attributes)
print(dt.check(tre3, m.monk3), dt.check(tre3, m.monk3test))
# draw.drawTree(tre3)






