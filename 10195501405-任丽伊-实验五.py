import torch
from torch import nn
from torchvision import models, transforms
import numpy as np
import os
import cv2
from sklearn.feature_extraction.text import TfidfVectorizer

model = models.vgg16(pretrained=True)
print(model)

class FeatureExtractor(nn.Module):
  def __init__(self, model):
    super(FeatureExtractor, self).__init__()
    self.features = list(model.features)
    self.features = nn.Sequential(*self.features)
    self.pooling = model.avgpool
    self.flatten = nn.Flatten()
    self.fc = model.classifier[0]
  
  def forward(self, x):
    out = self.features(x)
    out = self.pooling(out)
    out = self.flatten(out)
    out = self.fc(out) 
    return out 
 
model = models.vgg16(pretrained=True)
new_model = FeatureExtractor(model)
 
device = torch.device("cpu")
new_model = new_model.to(device)
print(new_model)

 
transform = transforms.Compose([
  transforms.ToPILImage(),
  transforms.CenterCrop(512),
  transforms.Resize(448),
  transforms.ToTensor()                              
])
 
features = []
 
f = open("train.txt","r")   #设置文件对象
str = f.read().split('\n')     #将txt文件的所有内容读入到字符串str中
f.close()   #将文件关闭
#print(len(str))
train_label = []
text_train = []
image_train = []
x_train = []
for i in range(1, len(str)-1):
  temp = str[i].split(",")
  path = os.path.join('data', temp[0] + '.jpg')
  img = cv2.imread(path)
  img = transform(img)
  img = img.reshape(1, 3, 448, 448)
  img = img.to(device)
  with torch.no_grad():
    feature = new_model(img)
  features.append(feature.cpu().detach().numpy().reshape(-1))
 
features = np.array(features)

features_test = []

f = open("test_without_label.txt","r")   #设置文件对象
str = f.read().split('\n')     #将txt文件的所有内容读入到字符串str中
f.close()   #将文件关闭
image_test = []
for i in range(1, len(str)-1):
    temp = str[i].split(",")
    path = os.path.join('data', temp[0] + '.jpg')
    img = cv2.imread(path)
    img = transform(img)
    img = img.reshape(1, 3, 448, 448)
    img = img.to(device)
    with torch.no_grad():
        feature_test = new_model(img)
    features_test.append(feature_test.cpu().detach().numpy().reshape(-1))
    
features_test = np.array(features_test)


f = open("test_without_label.txt","r")   #设置文件对象
str = f.read().split('\n')     #将txt文件的所有内容读入到字符串str中
f.close()   #将文件关闭
test_x = []
text_test = []
image_test = []
for i in range(1, len(str)-1):
    temp = str[i].split(",")
    # print(temp[0],temp[1])
    test_x.append(temp[0])
    text_path = 'data/' + temp[0] + '.txt'
    text = open(text_path, 'r', encoding='gb18030')
    text_test.append(text.read())
    text.close()

f = open("train.txt","r")   #设置文件对象
str = f.read().split('\n')     #将txt文件的所有内容读入到字符串str中
f.close()   #将文件关闭
#print(len(str))
train_label = []
text_train = []
image_train = []
x_train = []
# high, width = 0, 0 
for i in range(1, len(str)-1):
    # x_train.append(temp[0])
    temp = str[i].split(",")
    #print(temp[0],temp[1])
    train_label.append(temp[1])
    text_path = 'data/' + temp[0] + '.txt'
    text = open(text_path, 'r', encoding='gb18030')
    text_train.append(text.read())
    text.close()


tfidf = TfidfVectorizer()

features_text = tfidf.fit_transform(text_train+text_test).toarray()

text_train = features_text[:len(text_train)]
text_test = features_text[-len(text_test):]

print(features_text.shape, text_train.shape, text_test.shape)


image_train = features
combined_x_train = np.concatenate((np.array(image_train), np.array(text_train)), axis=1)
image_test = features_test
combined_x_test = np.concatenate((np.array(image_test), np.array(text_test)), axis=1)

print(combined_x_train.shape, combined_x_test.shape)

emotion_dict = {'positive': 0, 'neutral': 1, 'negative': 2}
for i in range(len(train_label)):
    train_label[i] = emotion_dict[train_label[i]]


import random
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

layer_acc = [0 for i in range(8)]
# 将数据每次都随机划分，然后随即划分5次，看这五次的平均准确率
for fold in range(2):
    # 将训练数据划分为训练集和数据集
    rand_factor = random.randrange(0,50) # 保证每一重验证时训练集和验证集划分的随机性
    # 将测试集和验证集按照0.75:0.25的比例分配
    # 尽量使验证集和测试集的大小相同
    x_train, x_verify, y_train, y_verify = train_test_split(combined_x_train,train_label,test_size=0.25,random_state=rand_factor)
    # 对几千个数据的大数据集solver‘adam'效果比较好
    layer = [5, 10, 20, 30, 50, 100, 150, 200]
    for i in range(8):
        classifier = MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(layer[i], ),max_iter = 10000, random_state=20)
        fit = classifier.fit(x_train,y_train)
        layer_acc[i] = layer_acc[i] + classifier.score(x_verify,y_verify)

for i in range(8):
    print('layer = ', end='')
    print(layer[i],end='  ')
    print(layer_acc[i]/2)

# 调整好参数之后利用所有的训练数据再次对模型进行训练
x_train, x_verify, y_train, y_verify = train_test_split(combined_x_train,train_label,test_size=0.25,random_state=rand_factor)
classifier = MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(150, ),max_iter = 10000, random_state=20)
fit = classifier.fit(x_train,y_train)

y_hat = classifier.predict(combined_x_test)

# emotion_dict = {'positive': 0, 'neutral': 1, 'negative': 2}
# 写入结果文件
f = open('test_without_label.txt',mode = 'w')
y_pred = []
for i in range(len(y_hat)):
    if y_hat[i] == 0: y_pred.append('positive')
    elif y_hat[i] == 1: y_pred.append('neutral')
    elif y_hat[i] == 2: y_pred.append('negative')
f.write( 'guid, tag' + '\n')
for i in range(len(y_hat)):
    # print(type(i))
    temp = y_pred[i]
    f.write(test_x[i] + ',' + temp + '\n')

f.close()


import random
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

layer_acc = [0 for i in range(8)]
# 将数据每次都随机划分，然后随即划分5次，看这五次的平均准确率
for fold in range(2):
    # 将训练数据划分为训练集和数据集
    rand_factor = random.randrange(0,50) # 保证每一重验证时训练集和验证集划分的随机性
    # 将测试集和验证集按照0.75:0.25的比例分配
    # 尽量使验证集和测试集的大小相同
    x_train, x_verify, y_train, y_verify = train_test_split(np.array(image_train),train_label,test_size=0.25,random_state=rand_factor)
    # 对几千个数据的大数据集solver‘adam'效果比较好
    layer = [5, 10, 20, 30, 50, 100, 150, 200]
    for i in range(8):
        classifier = MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(layer[i], ),max_iter = 10000, random_state=20)
        fit = classifier.fit(x_train,y_train)
        layer_acc[i] = layer_acc[i] + classifier.score(x_verify,y_verify)


for i in range(8):
    print('layer = ', end='')
    print(layer[i],end='  ')
    print(layer_acc[i]/2)

import random
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

layer_acc = [0 for i in range(8)]
# 将数据每次都随机划分，然后随即划分5次，看这五次的平均准确率
for fold in range(2):
    # 将训练数据划分为训练集和数据集
    rand_factor = random.randrange(0,50) # 保证每一重验证时训练集和验证集划分的随机性
    # 将测试集和验证集按照0.75:0.25的比例分配
    # 尽量使验证集和测试集的大小相同
    x_train, x_verify, y_train, y_verify = train_test_split(np.array(text_train),train_label,test_size=0.25,random_state=rand_factor)
    # 对几千个数据的大数据集solver‘adam'效果比较好
    layer = [5, 10, 20, 30, 50, 100, 150, 200]
    for i in range(8):
        classifier = MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(layer[i], ),max_iter = 10000, random_state=20)
        fit = classifier.fit(x_train,y_train)
        layer_acc[i] = layer_acc[i] + classifier.score(x_verify,y_verify)

for i in range(8):
    print('layer = ', end='')
    print(layer[i],end='  ')
    print(layer_acc[i]/2)