
# coding: utf-8

# # Загрузка данных

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


# In[2]:


kk=100 #размер тренировочного набора
path_cropped ='/home/ann/Documents/colors/cropped/' # путь к папке, где находятся фрагменты,из которых создается изображение
path_saved_txt = "/home/ann/Documents/colors/txt/" # путь к папке, куда записываются созданные текстовые файлы с разметкой
path_saved_png = "/home/ann/Documents/colors/png/" # путь к папке, где хранятся созданные изображения
path_test_png = '/home/ann/Документы/colors/pngtest/' # изображения из тестового набора
path_test_txt = '/home/ann/Документы/colors/txttest/' # разметка для тестовых изображений


# In[3]:


d = {0: "Розовый", 1: "Красный", 2:"Салатовый",
     3:"Желтый", 4: "Бирюзовый",
     5:"Сиреневый",6:"Синий", 
     7:"Черный",8:"Белый",
     9:"Оранжевый",10:"Фиолетовый",
     11:"Темно-зеленый",12:"Зеленый"     
}


# In[4]:


x_train=mpimg.imread(path_saved_png+str(0)+'.png').reshape(1,256,256,3)
for glob in range(1,kk):
    img2=mpimg.imread(path_saved_png+str(glob)+'.png').reshape(1,256,256,3)
    #print(glob, end=" ")
    x_train=np.vstack((x_train,img2))


# In[5]:


print(x_train.shape)


# In[6]:


a=[]


# In[7]:


for o in range(64):
    pos=[]
    for glob in range(kk):
        with open(path_saved_txt+str(glob)+".txt") as file:
            numbers=[]
            for line in file:
                for num in line.split():
                    numbers.append(int(num))
        pos.append(numbers[o])
    y_train=np.array([pos])    
    #print(y_train.shape)
    y_train=y_train.reshape(1,kk)
    a.append(y_train)


# In[9]:


from keras.utils import np_utils
for h in range(64):
    a[h]=np_utils.to_categorical(a[h],13)
for g in range(64):
    a[g]=a[g].reshape(kk,13)


# In[10]:


#Проверка соответствия лейблов и картинок
index = 2
import matplotlib.pyplot as plt
for s in range(64):
    print(s, d[ np.argmax(a[s][index])])

plt.imshow(x_train[index])
plt.show()
#print(x_train[index])


# # Модель

# In[11]:


from keras.models import Model 
from keras.layers import Input, Dense, Flatten, Convolution2D, MaxPooling2D, Dropout, merge
from keras import optimizers


# In[12]:


kernel_size = 3 
pool_size = 2
conv_depth = 32
conv_depth1 = 64
drop_prob_1 = 0.25 
drop_prob_2 = 0.5 
hidden_size = 128


# In[13]:


inp = Input(shape=(256, 256, 3))
conv_1 = Convolution2D(conv_depth, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
conv_2 = Convolution2D(conv_depth, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size),data_format=None)(conv_2)
drop_1 = Dropout(drop_prob_1)(pool_1)
conv_3 = Convolution2D(conv_depth1, (kernel_size, kernel_size),padding='same', activation='relu')(drop_1)
conv_4 = Convolution2D(conv_depth1, (kernel_size, kernel_size),padding='same', activation='relu')(conv_3)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size),data_format=None)(conv_4)
drop_2 = Dropout(drop_prob_1)(pool_2)
flat = Flatten()(drop_2)
hidden = Dense(hidden_size, activation='relu')(flat)
drop_3 = Dropout(drop_prob_2)(hidden)
out1 = Dense(13, activation='softmax')(drop_3)
out2 = Dense(13, activation='softmax')(drop_3)
out3 = Dense(13, activation='softmax')(drop_3)
out4 = Dense(13, activation='softmax')(drop_3)
out5 = Dense(13, activation='softmax')(drop_3)
out6 = Dense(13, activation='softmax')(drop_3)
out7 = Dense(13, activation='softmax')(drop_3)
out8 = Dense(13, activation='softmax')(drop_3)
out9 = Dense(13, activation='softmax')(drop_3)
out10 = Dense(13, activation='softmax')(drop_3)
out11 = Dense(13, activation='softmax')(drop_3)
out12 = Dense(13, activation='softmax')(drop_3)
out13 = Dense(13, activation='softmax')(drop_3)
out14 = Dense(13, activation='softmax')(drop_3)
out15 = Dense(13, activation='softmax')(drop_3)
out16 = Dense(13, activation='softmax')(drop_3)
out17 = Dense(13, activation='softmax')(drop_3)
out18 = Dense(13, activation='softmax')(drop_3)
out19 = Dense(13, activation='softmax')(drop_3)
out20 = Dense(13, activation='softmax')(drop_3)
out21 = Dense(13, activation='softmax')(drop_3)
out22 = Dense(13, activation='softmax')(drop_3)
out23 = Dense(13, activation='softmax')(drop_3)
out24 = Dense(13, activation='softmax')(drop_3)
out25 = Dense(13, activation='softmax')(drop_3)
out26 = Dense(13, activation='softmax')(drop_3)
out27 = Dense(13, activation='softmax')(drop_3)
out28 = Dense(13, activation='softmax')(drop_3)
out29 = Dense(13, activation='softmax')(drop_3)
out30 = Dense(13, activation='softmax')(drop_3)
out31 = Dense(13, activation='softmax')(drop_3)
out32 = Dense(13, activation='softmax')(drop_3)
out33 = Dense(13, activation='softmax')(drop_3)
out34 = Dense(13, activation='softmax')(drop_3)
out35 = Dense(13, activation='softmax')(drop_3)
out36 = Dense(13, activation='softmax')(drop_3)
out37 = Dense(13, activation='softmax')(drop_3)
out38 = Dense(13, activation='softmax')(drop_3)
out39 = Dense(13, activation='softmax')(drop_3)
out40 = Dense(13, activation='softmax')(drop_3)
out41 = Dense(13, activation='softmax')(drop_3)
out42 = Dense(13, activation='softmax')(drop_3)
out43 = Dense(13, activation='softmax')(drop_3)
out44 = Dense(13, activation='softmax')(drop_3)
out45 = Dense(13, activation='softmax')(drop_3)
out46 = Dense(13, activation='softmax')(drop_3)
out47 = Dense(13, activation='softmax')(drop_3)
out48 = Dense(13, activation='softmax')(drop_3)
out49 = Dense(13, activation='softmax')(drop_3)
out50 = Dense(13, activation='softmax')(drop_3)
out51 = Dense(13, activation='softmax')(drop_3)
out52 = Dense(13, activation='softmax')(drop_3)
out53 = Dense(13, activation='softmax')(drop_3)
out54 = Dense(13, activation='softmax')(drop_3)
out55 = Dense(13, activation='softmax')(drop_3)
out56 = Dense(13, activation='softmax')(drop_3)
out57 = Dense(13, activation='softmax')(drop_3)
out58 = Dense(13, activation='softmax')(drop_3)
out59 = Dense(13, activation='softmax')(drop_3)
out60 = Dense(13, activation='softmax')(drop_3)
out61 = Dense(13, activation='softmax')(drop_3)
out62 = Dense(13, activation='softmax')(drop_3)
out63 = Dense(13, activation='softmax')(drop_3)
out64 = Dense(13, activation='softmax')(drop_3)


# In[14]:


model = Model(inputs=inp,
              outputs=[out1, out2,out3,out4, out5,out6,out7, out8,out9,out10,
                      out11, out12,out13,out14, out15,out16,out17, out18,out19,out20,
                      out21, out22,out23,out24, out25,out26,out27, out28,out29,out30,
                      out31, out32,out33,out34, out35,out36,out37, out38,out39,out40,
                      out41, out42,out43,out44, out45,out46,out47, out48,out49,out50,
                      out51, out52,out53,out54, out55,out56,out57, out58,out59,out60,
                      out61, out62,out63,out64])


# In[15]:


model.compile(loss='categorical_crossentropy',
              optimizer = optimizers.SGD(lr=0.0025),
              metrics=['accuracy'])


# In[16]:


model.fit(x_train, [a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],
                    a[10],a[11],a[12],a[13],a[14],a[15],a[16],a[17],a[18],a[19],
                    a[20],a[21],a[22],a[23],a[24],a[25],a[26],a[27],a[28],a[29],
                    a[30],a[31],a[32],a[33],a[34],a[35],a[36],a[37],a[38],a[39],
                    a[40],a[41],a[42],a[43],a[44],a[45],a[46],a[47],a[48],a[49],
                    a[50],a[51],a[52],a[53],a[54],a[55],a[56],a[57],a[58],a[59],
                    a[60],a[61],a[62],a[63]],
          batch_size=12, epochs=500,
          verbose=1, validation_split=0.1)


# In[ ]:


scores = model.evaluate(x_train, [a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],
                    a[10],a[11],a[12],a[13],a[14],a[15],a[16],a[17],a[18],a[19],
                    a[20],a[21],a[22],a[23],a[24],a[25],a[26],a[27],a[28],a[29],
                    a[30],a[31],a[32],a[33],a[34],a[35],a[36],a[37],a[38],a[39],
                    a[40],a[41],a[42],a[43],a[44],a[45],a[46],a[47],a[48],a[49],
                    a[50],a[51],a[52],a[53],a[54],a[55],a[56],a[57],a[58],a[59],
                    a[60],a[61],a[62],a[63]], verbose=1)
print(scores)


# In[ ]:


#Проверка работы модели на примере из тренировочного набора
num=45
x=mpimg.imread(path_saved_png+str(num)+'.png').reshape(1,256,256,3)
data = []
answer=[]
with open(path_saved_txt+str(num)+'.txt') as f:
    for line in f:
        data.append([int(x) for x in line.split()])
for i in range(8):
    for k in range(8):
        answer.append(data[i][k])    
corr=0    
    
prediction = model.predict(x)
for i in range(64):
    pr=prediction[i][0].max()
    itemindex = np.where(prediction[i][0]==pr)
    print(d[itemindex[0][0]])
    print(itemindex[0][0])
    if (itemindex[0][0]==answer[i]):
        corr+=1
        
print(corr,'/64')   
x=x.reshape(256,256,3)
import matplotlib.pyplot as plt
plt.imshow(x)
plt.show()


# # Загрузка данных для теста

# In[ ]:


nn=500
x_test=mpimg.imread(path_test_png+str(0)+'.png').reshape(1,256,256,3)

for glob in range(1,nn):
    img2=mpimg.imread(path_test_png+str(glob)+'.png').reshape(1,256,256,3)
    print(glob, end=" ")
    x_test=np.vstack((x_test,img2))


# In[ ]:


b=[]
koli=64
for o in range(koli):
    posi=[]
    for glob in range(nn):
        with open(path_test_txt+str(glob)+".txt") as file:
            number=[]
            for line in file:
                for num in line.split():
                    number.append(int(num))
        posi.append(number[o])
    y_test=np.array([posi])    
    print(y_test.shape)
    y_test=y_test.reshape(1,nn)
    b.append(y_test)
for h in range(koli):
    b[h]=np_utils.to_categorical(b[h],13)
for g in range(koli):
    b[g]=b[g].reshape(nn,13)


# In[ ]:


#Проверка соответствия лейблов картинкам
index = 2
import matplotlib.pyplot as plt
for s in range(koli):
    print(s, d[ np.argmax(b[s][index])])

plt.imshow(x_test[index])
plt.show()
print(x_test[index])


# In[ ]:


scores = model.evaluate(x_test, [b[0],b[1],b[2],b[3],b[4],b[5],b[6], b[7],b[8],b[9],
                    b[10],b[11],b[12],b[13],b[14],b[15],b[16],b[17],b[18],b[19],
                    b[20],b[21],b[22],b[23],b[24],b[25],b[26],b[27],b[28],b[29],
                    b[30],b[31],b[32],b[33],b[34],b[35],b[36],b[37],b[38],b[39],
                    b[40],b[41],b[42],b[43],b[44],b[45],b[46],b[47],b[48],b[49],
                    b[50],b[51],b[52],b[53],b[54],b[55],b[56],b[57],b[58],b[59],
                    b[60],b[61],b[62],b[63]], verbose=1)
print(scores)


# model_json = model.to_json()

# json_file = open("colors_model64.json", "w")
# json_file.write(model_json)
# json_file.close()

# model.save_weights("colors_model64.h5")

# from keras.models import model_from_json
# model_json = model.to_json()
# with open("colmodel64.json", "w") as json_file:
#     json_file.write(model_json)
# 
