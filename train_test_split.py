import glob
import os
import shutil

folders = glob.glob('./data/**/')
folders

os.makedirs('data/train_data/')
os.makedirs('data/test_data/')

# take the first 80 random folders given by glob
train_folder = folders[:160]

for train in train_folder:
    shutil.move(train,'data/train_data/')

test_folder = glob.glob('./data/**/')
test_folder = test_folder[2:]
test_folder

for test in test_folder:
    shutil.move(test,'data/test_data/')

parameters = glob.glob('./data/*.csv')
check=[]
for name in parameters:
    name = name.split('/')[2]
    name = name.split('_')[0]
    check.append(name)
check


# In[18]:


train_folder


# In[23]:


numbers = []
for number in train_folder:
    x = number.split('/',-1)
    numbers.append(x[2])


# In[24]:


numbers


# In[34]:


for no in numbers :
    shutil.move('./data/'+no+'_parameters.csv', './data/test_data/')


# In[ ]:




