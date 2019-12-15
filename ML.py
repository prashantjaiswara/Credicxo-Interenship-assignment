#!/usr/bin/env python
# coding: utf-8

# In[135]:


import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
import matplotlib.pyplot as plt
import joblib


# In[136]:


chemical_data=pd.read_csv('musk_csv.csv')
#chemical_data

X=chemical_data.drop(labels=['ID','molecule_name','conformation_name','class'],axis=1)
#X

Y=chemical_data['class']
#Y


# In[137]:


X_for_training, X_for_testing, Y_for_training, Y_for_testing = train_test_split(X,Y,test_size=0.2,random_state=0)


# In[138]:


model=MLPClassifier(activation='logistic', solver='sgd',hidden_layer_sizes=(10,15),max_iter=1000, random_state=0)


# In[139]:


model.fit(X_for_training, Y_for_training)


# In[140]:


Y_predicted=model.predict(X_for_testing)


# In[141]:


str=classification_report(Y_for_testing, Y_predicted)
file=open("Final Performace Measures.txt","a")
file.write(str)
file.close()

print(str)


# In[142]:


model_accuracy_score=accuracy_score(Y_for_testing, Y_predicted)
model_accuracy_score


# In[154]:


loss_values = model.loss_curve_
plt.plot(loss_values,'-b',label='train')
plt.xlabel("n iteration")
plt.legend(loc='upper left')
plt.title("loss with iteration")

plt.show()



# In[143]:


joblib.dump(model,'chemical_type_predictor_MLP.joblib')


# In[144]:
retrived_model=joblib.load('chemical_type_predictor.joblib')

input_data_1=[50,-196,-149,28,-117,-81,58,-4,-39,-119,-44,101,-51,-57,-72,-296,43,-82,129,-26,-6,-17,1,-54,-98,-14,29,-76,12,-147,-116,-103,-15,-227,-34,22,-167,36,-146,13,-178,-86,-72,-61,36,-108,-109,-64,90,-98,58,-17,-6,72,-51,-57,85,-142,-15,-165,-20,-195,31,-191,-11,48,-166,-58,-133,-74,-163,-150,-44,51,-61,-184,-17,-59,121,62,-1,14,67,-28,-180,35,-109,-105,29,-149,-202,-11,3,-119,16,-47,-77,40,-153,66,-199,20,-54,-52,-63,-120,-179,-84,-67,33,1,19,28,11,-150,-18,-92,55,86,14,-117,-67,-101,-62,-128,19,-145,-188,-116,-46,-15,-118,-50,41,-63,-49,74,-45,-45,-19,63,5,40,24,-178,-103,-118,-92,-103,29,-22,3,8,-170,-70,-196,-241,-254,-212,-119,-10,33,187,-70,-167,48]
input_data_2=[114,61,-144,-77,-117,11,56,-165,-40,-22,-293,-113,-67,-275,-284,-303,52,-154,-98,-184,-29,-22,2,104,111,-34,48,-87,25,68,-114,157,-30,12,-128,211,-175,3,-128,39,-93,161,-27,-297,-244,-327,-98,-72,-107,-102,-73,-26,-12,-121,148,-93,93,-119,35,41,-101,178,21,-5,-146,179,-166,-84,-118,-55,-139,-82,-164,-193,-218,-182,1,-282,-105,-166,-20,13,-32,-39,1,23,-43,-167,-8,82,114,166,-115,-48,30,-55,-167,31,-156,80,-28,266,-257,-284,-292,-265,-167,-120,-245,-248,-232,2,-3,-3,-1,61,112,37,84,25,81,149,130,-108,96,6,41,56,14,-188,25,-138,-75,-127,-221,-93,53,-70,-18,-33,-27,5,49,17,-178,-102,-119,-57,-52,53,-62,-122,-113,83,180,151,-239,-81,-137,34,254,290,143,-58,-117,60]



Y_predicted=retrived_model.predict( [input_data_1, input_data_2])
print("Output:",Y_predicted)

