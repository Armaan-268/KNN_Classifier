import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset_x = pd.read_csv('Diabetes_XTrain.csv')
dataset_y = pd.read_csv('Diabetes_YTrain.csv')

# Plot_1: Pregnancy(Weeks) vs Diabetes
x = dataset_x.iloc[1:, 0].values
y = dataset_y.iloc[1:, 0].values
prg = []
no = []
dib = []
for i in x:
    if i not in prg:
        prg.append(i)
prg.sort()
for i in prg:
    count = 0
    for j in x:
        if i == j:
            count = count + 1
    no.append(count)
for i in prg:
    count = 0
    for j in range (0,len(x)-1):
        if x[j] == i and y[j]==1:
            count = count + 1
    dib.append(count)
ind = np.arange(len(prg)) 
width = 0.35
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(ind + 0.00, no, color = 'b', width = 0.25)
ax.bar(ind + 0.25, dib, color = 'r', width = 0.25)
plt.xticks(ind,prg)
ax.legend(labels=['Total', 'Diebetic'])
plt.title('Pregnancy(Weeks) vs Diabetes')
plt.xlabel('Pregnancy(Weeks)')
plt.ylabel('Number')
plt.show()


# Plot_2: Glucose vs Diabetes
x = dataset_x.iloc[1:, 1].values
y = dataset_y.iloc[1:, 0].values
prg = []

no = []
dib = []
for i in x:
    if i not in prg:
        prg.append(i)
prg.sort()
for i in prg:
    count = 0
    for j in x:
        if i == j:
            count = count + 1
    no.append(count)
for i in prg:
    count = 0
    for j in range (0,len(x)-1):
        if x[j] == i and y[j]==1:
            count = count + 1
    dib.append(count)
ind = np.arange(len(prg)) 
width = 0.35
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(ind + 0.00, no, color = 'b', width = 0.25)
ax.bar(ind + 0.25, dib, color = 'r', width = 0.25)
plt.xticks(ind,prg)
ax.legend(labels=['Total', 'Diebetic'])
plt.title('Glucose Level vs Diabetes')
plt.xlabel('Glucose')
plt.ylabel('Number')
plt.show()

# Plot_3: Blood Pressure vs Diabetes
x = dataset_x.iloc[1:, 2].values
y = dataset_y.iloc[1:, 0].values
prg = []
no = []
dib = []
for i in x:
    if i not in prg:
        prg.append(i)
prg.sort()
for i in prg:
    count = 0
    for j in x:
        if i == j:
            count = count + 1
    no.append(count)
for i in prg:
    count = 0
    for j in range (0,len(x)-1):
        if x[j] == i and y[j]==1:
            count = count + 1
    dib.append(count)
N = 0
for i in prg:
    if i>N:
        N = i
ind = np.arange(len(prg)) 
width = 2
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(ind + 0.00, no, color = 'b', width = 0.25)
ax.bar(ind + 0.25, dib, color = 'r', width = 0.25)
plt.xticks(ind,prg)
ax.legend(labels=['Total', 'Diebetic'])
plt.title('Blood Pressure vs Diabetes')
plt.xlabel('Blood Pressure')
plt.ylabel('Number')
plt.show()

# Plot_4: SkinThickness vs Diabetes
x = dataset_x.iloc[1:, 3].values
y = dataset_y.iloc[1:, 0].values
prg = []
no = []
dib = []
for i in x:
    if i not in prg:
        prg.append(i)
prg.sort()
for i in prg:
    count = 0
    for j in x:
        if i == j:
            count = count + 1
    no.append(count)
for i in prg:
    count = 0
    for j in range (0,len(x)-1):
        if x[j] == i and y[j]==1:
            count = count + 1
    dib.append(count)
N = 0
for i in prg:
    if i>N:
        N = i
ind = np.arange(len(prg)) 
width = 2
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(ind + 0.00, no, color = 'b', width = 0.25)
ax.bar(ind + 0.25, dib, color = 'r', width = 0.25)
plt.xticks(ind,prg)
ax.legend(labels=['Total', 'Diebetic'])
plt.title('SkinThickness vs Diabetes')
plt.xlabel('SkinThickness')
plt.ylabel('Number')
plt.show()


# Plot_5: Insuline vs Diabetes
x = dataset_x.iloc[1:, 4].values
y = dataset_y.iloc[1:, 0].values
prg = []
no = []
dib = []
for i in x:
    if i not in prg:
        prg.append(i)
prg.sort()
for i in prg:
    count = 0
    for j in x:
        if i == j:
            count = count + 1
    no.append(count)
for i in prg:
    count = 0
    for j in range (0,len(x)-1):
        if x[j] == i and y[j]==1:
            count = count + 1
    dib.append(count)
N = 0
for i in prg:
    if i>N:
        N = i
ind = np.arange(len(prg)) 
width = 2
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(ind + 0.00, no, color = 'b', width = 0.25)
ax.bar(ind + 0.25, dib, color = 'r', width = 0.25)
plt.xticks(ind,prg)
ax.legend(labels=['Total', 'Diebetic'])
plt.title('Insuline vs Diabetes')
plt.xlabel('Insuline')
plt.ylabel('Number')
plt.show()

# Plot_6: BMI vs Diabetes
x = dataset_x.iloc[1:, 5].values
y = dataset_y.iloc[1:, 0].values
prg = []
no = []
dib = []
for i in x:
    if i not in prg:
        prg.append(i)
prg.sort()
for i in prg:
    count = 0
    for j in x:
        if i == j:
            count = count + 1
    no.append(count)
for i in prg:
    count = 0
    for j in range (0,len(x)-1):
        if x[j] == i and y[j]==1:
            count = count + 1
    dib.append(count)
N = 0
for i in prg:
    if i>N:
        N = i
ind = np.arange(len(prg)) 
width = 2
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(ind + 0.00, no, color = 'b', width = 0.25)
ax.bar(ind + 0.25, dib, color = 'r', width = 0.25)
plt.xticks(ind,prg)
ax.legend(labels=['Total', 'Diebetic'])
plt.title('BMI vs Diabetes')
plt.xlabel('BMI')
plt.ylabel('Number')
plt.show()

# Plot_7: DPF vs Diabetes
x = dataset_x.iloc[1:, 6].values
y = dataset_y.iloc[1:, 0].values
prg = []
no = []
dib = []
for i in x:
    if i not in prg:
        prg.append(i)
prg.sort()
for i in prg:
    count = 0
    for j in x:
        if i == j:
            count = count + 1
    no.append(count)
for i in prg:
    count = 0
    for j in range (0,len(x)-1):
        if x[j] == i and y[j]==1:
            count = count + 1
    dib.append(count)
N = 0
for i in prg:
    if i>N:
        N = i
ind = np.arange(len(prg)) 
width = 2
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(ind + 0.00, no, color = 'b', width = 0.25)
ax.bar(ind + 0.25, dib, color = 'r', width = 0.25)
plt.xticks(ind,prg)
ax.legend(labels=['Total', 'Diebetic'])
plt.title('DPF vs Diabetes')
plt.xlabel('DPF')
plt.ylabel('Number')
plt.show()

# Plot_8: AGE vs Diabetes
x = dataset_x.iloc[1:, 7].values
y = dataset_y.iloc[1:, 0].values
prg = []
no = []
dib = []
for i in x:
    if i not in prg:
        prg.append(i)
prg.sort()
for i in prg:
    count = 0
    for j in x:
        if i == j:
            count = count + 1
    no.append(count)
for i in prg:
    count = 0
    for j in range (0,len(x)-1):
        if x[j] == i and y[j]==1:
            count = count + 1
    dib.append(count)
N = 0
for i in prg:
    if i>N:
        N = i
ind = np.arange(len(prg)) 
width = 2
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(ind + 0.00, no, color = 'b', width = 0.25)
ax.bar(ind + 0.25, dib, color = 'r', width = 0.25)
plt.xticks(ind,prg)
ax.legend(labels=['Total', 'Diebetic'])
plt.title('Age vs Diabetes')
plt.xlabel('AGE')
plt.ylabel('Number')
plt.show()
