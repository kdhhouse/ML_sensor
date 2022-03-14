import os
import csv
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

model = torch.load('model_epoch_1000_100_2.pth', map_location=torch.device('cpu'))

f = open('Type3_test_rand.csv', 'r', encoding='utf-8')
g = open('Type3_train_2000.csv', 'r', encoding='utf-8')
testset = csv.reader(f)
trainset = csv.reader(g)
input_data = []
label_data = []
for line in testset:
    count = 0
    input_content = []
    label_content = []
    for content in line:
        if 21 < count < 28:
            input_content.append(float(content) * 100)
        elif count < 6:
            label_content.append(float(content)*0.1)

        else:
            pass
        count += 1
    input_data.append(input_content)
    label_data.append(label_content)

# input_data = np.array(input_data)+0.001*np.random.rand(np.shape(input_data)[0],np.shape(input_data)[1])
input_data = np.array(input_data)

label_data = np.array(label_data)*[10, 10, 10, 0.2, 0.2, 0.2]*0.1

f.close()

new_var = torch.FloatTensor(input_data)
pred = model(new_var).detach().numpy()
pred_y = np.zeros(np.shape(label_data))

for i in range (1,np.shape(label_data)[0]):
    pred_y[i-1] = ((pred[i-1])-label_data[i-1])
    # pred_y[i - 1] = ((pred[i - 1]))
f.close()
print(np.max(np.abs(pred_y)))
print(pred)
# df = pd.DataFrame(pred)
# df.to_csv('height_transfer.csv', index=False)
