#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import warnings
warnings.simplefilter("ignore")


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


from skimage.io import imread,imshow
from skimage.transform import resize
from skimage.color import rgb2gray


# In[5]:


priya=os.listdir("C:/Users/kiran vignesh/Downloads/priya")


# In[6]:


kidd=os.listdir("C:/Users/kiran vignesh/Downloads/kidd")
ed=os.listdir("C:/Users/kiran vignesh/Downloads/ed")


# In[7]:


limit=10
priya_images=[None]*limit
j=0
for i in priya:
    if(j<limit):
        priya_images[j]=imread("C:/Users/kiran vignesh/Downloads/priya/"+i)
        j+=1
    else:
        break


# In[8]:


imshow(priya_images[0])


# In[9]:


limit=10
kidd_images=[None]*limit
j=0
for i in kidd:
    if(j<limit):
        kidd_images[j]=imread("C:/Users/kiran vignesh/Downloads/kidd/"+i)
        j+=1
    else:
        break


# In[10]:


imshow(kidd_images[0])


# In[11]:


limit=10
ed_images=[None]*limit
j=0
for i in ed:
    if(j<limit):
        ed_images[j]=imread("C:/Users/kiran vignesh/Downloads/ed/"+i)
        j+=1
    else:
        break


# In[12]:


imshow(ed_images[0])


# In[13]:


priya_gray=[None]*limit
j=0
for i in priya:
    if(j<limit):
        priya_gray[j]=rgb2gray(priya_images[j])
        j+=1
    else:
        break


# In[14]:


kidd_gray=[None]*limit
j=0
for i in kidd:
    if(j<limit):
        kidd_gray[j]=rgb2gray(kidd_images[j])
        j+=1
    else:
        break


# In[15]:


ed_gray=[None]*limit
j=0
for i in ed:
    if(j<limit):
        ed_gray[j]=rgb2gray(ed_images[j])
        j+=1
    else:
        break


# In[16]:


imshow(priya_gray[0])


# In[17]:


imshow(kidd_gray[0])


# In[18]:


imshow(ed_gray[0])


# In[19]:


priya_gray[3].shape


# In[20]:


kidd_gray[3].shape


# In[21]:


ed_gray[3].shape


# In[22]:


for j in range(10):
  pi=priya_gray[j]
  priya_gray[j]=resize(pi,(512,512))


# In[23]:


for j in range(10):
  ki=kidd_gray[j]
  kidd_gray[j]=resize(ki,(512,512))


# In[24]:


for j in range(10):
  ed=ed_gray[j]
  ed_gray[j]=resize(ed,(512,512))


# In[25]:


imshow(priya_gray[4])


# In[26]:


imshow(kidd_gray[4])


# In[27]:


imshow(ed_gray[4])


# In[28]:


len_of_images_priya=len(priya_gray)
len_of_images_priya


# In[29]:


len_of_images_kidd=len(kidd_gray)
len_of_images_kidd


# In[30]:


len_of_images_ed=len(ed_gray)
len_of_images_ed


# In[31]:


image_size_priya=priya_gray[1].shape
image_size_priya


# In[32]:


image_size_kidd=kidd_gray[1].shape
image_size_kidd


# In[33]:


image_size_ed=ed_gray[1].shape
image_size_ed


# In[34]:


flatten_size_priya=image_size_priya[0]*image_size_priya[1]
flatten_size_priya


# In[35]:


flatten_size_kidd=image_size_kidd[0]*image_size_kidd[1]
flatten_size_kidd


# In[36]:


flatten_size_ed=image_size_ed[0]*image_size_ed[1]
flatten_size_ed


# In[37]:


for i in range(len_of_images_priya):
  priya_gray[i]=np.ndarray.flatten(priya_gray[i]).reshape(flatten_size_priya,1)


# In[38]:


for i in range(len_of_images_kidd):
  kidd_gray[i]=np.ndarray.flatten(kidd_gray[i]).reshape(flatten_size_kidd,1)


# In[39]:


priya_gray=np.dstack(priya_gray)
priya_gray.shape


# In[40]:


kidd_gray=np.dstack(kidd_gray)
kidd_gray.shape


# In[41]:


ed_gray=np.dstack(ed_gray)
ed_gray.shape


# In[42]:


priya_gray=np.rollaxis(priya_gray,axis=2,start=0)
priya_gray.shape


# In[43]:


kidd_gray=np.rollaxis(kidd_gray,axis=2,start=0)
kidd_gray.shape


# In[44]:


ed_gray=np.rollaxis(ed_gray,axis=2,start=0)
ed_gray.shape


# In[45]:


priya_gray=priya_gray.reshape(len_of_images_priya,flatten_size_priya)
priya_gray.shape


# In[46]:


kidd_gray=kidd_gray.reshape(len_of_images_kidd,flatten_size_kidd)
kidd_gray.shape


# In[47]:


ed_gray=ed_gray.reshape(len_of_images_ed,flatten_size_ed)
ed_gray.shape


# In[48]:


priya_data=pd.DataFrame(priya_gray)
priya_gray


# In[49]:


kidd_data=pd.DataFrame(kidd_gray)
kidd_gray


# In[50]:


ed_data=pd.DataFrame(ed_gray)
ed_gray


# In[51]:


priya_data["label"]="priya"
priya_data


# In[52]:


kidd_data["label"]="kidd"
kidd_data


# In[53]:


ed_data["label"]="ed"
ed_data


# In[54]:


actor_1=pd.concat([priya_data,ed_data])


# In[55]:


actor=pd.concat([actor_1,kidd_data])


# In[56]:


actor


# In[57]:


from sklearn.utils import shuffle
hollywood_indexed=shuffle(actor).reset_index()
hollywood_indexed


# In[58]:


hollywood_actors=hollywood_indexed.drop(["index"],axis=1)
hollywood_actors


# In[59]:


x=hollywood_actors.values[:,:-1]


# In[60]:


y=hollywood_actors.values[:,-1]


# In[61]:


x


# In[62]:


y


# In[63]:


from sklearn.model_selection import train_test_split


# In[64]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[65]:


from sklearn import svm


# In[66]:


clf=svm.SVC()
clf.fit(x_train,y_train)


# In[67]:


y_pred=clf.predict(x_test)


# In[68]:


y_pred


# In[69]:


for i in (np.random.randint(0,6,4)):
  predicted_images=(np.reshape(x_test[i],(512,512)).astype(np.float64))
  plt.title("Predicted label: {0}".format(y_pred[i]))
  plt.imshow(predicted_images,interpolation="nearest",cmap="gray")
  plt.show()


# In[70]:


from sklearn import metrics


# In[71]:


accuracy=metrics.accuracy_score(y_test,y_pred)


# In[72]:


accuracy


# In[73]:


from sklearn.metrics import confusion_matrix


# In[74]:


confusion_matrix(y_test,y_pred)


# In[ ]:





# In[ ]:





# In[ ]:




