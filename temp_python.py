import numpy as np

all_lines = open('temp_file.txt').readlines()
MAE = []
R2 = []
for line in all_lines:
    all_comp = line.strip().split()
    MAE.append(float(all_comp[1]))
    R2.append(float(all_comp[5]))
print("$",round(np.mean(R2),3),"\pm",round(np.std(R2), 3),"$ & $", round(np.mean(MAE),2),"\pm",round(np.std(MAE), 2),"$")
