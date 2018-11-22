
# coding: utf-8

# In[1]:


n = 1000 #размер датасета
path_cropped ='/home/ann/Documents/colors/cropped/' # путь к папке, где находятся фрагменты,из которых создается изображение
path_saved_txt = "/home/ann/Documents/colors/txt/" # путь к папке, куда записываются созданные текстовые файлы с разметкой
path_saved_png = "/home/ann/Documents/colors/png/" # путь к папке, где хранятся созданные изображения


# In[2]:


from PIL import Image
import random


# In[3]:


img = Image.new('RGB', (256, 256), 'white')

for glob in range(n):
    f = open(path_saved_txt+str(glob)+".txt", 'w')
    for j in range(8):
        l=[]
        for k in range(8):
            l.append(random.randint(0,12))
    
        for i in range(8):
            img.paste(Image.open(path_cropped+str(l[i])+'.png'),(i*32,j*32))
            f.write(str(l[i]) + '\t')
        f.write('\n')    
    img.save(path_saved_png+str(glob)+".png")
    f.close()

