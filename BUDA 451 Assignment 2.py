#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 4a
#Decision Tree Model

import pandas as pd

trainData = pd.read_csv('https://raw.githubusercontent.com/binbenliu/Teaching/main/data/Diabetes/diabetes_train.csv',header='infer')
trainData


# In[2]:


trainData.dtypes


# In[3]:


from sklearn import tree

Y = trainData['Outcome']
X = trainData.drop(['Outcome'],axis=1)

clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=None)
clf = clf.fit(X, Y)


# In[4]:


import pydotplus
from IPython.display import Image

dot_data = tree.export_graphviz(clf, feature_names=X.columns, class_names=['0','1'], filled=True,
                                out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

# The decision tree makes its classifications by starting with the glucose level and then looks at every attribute to see if the specific row is above or below the level of that certain attribute.


# In[5]:


testData = pd.read_csv('https://raw.githubusercontent.com/binbenliu/Teaching/main/data/Diabetes/diabetes_test.csv',header='infer')
testData


# In[6]:


testY = testData['Outcome']
testX = testData.drop(['Outcome'],axis=1)

predY = clf.predict(testX)
predictions = pd.concat([testData,pd.Series(predY,name='Predicted Outcome')], axis=1)
predictions


# In[7]:


from sklearn.metrics import accuracy_score

print('Accuracy on test data is %.2f' % (accuracy_score(testY, predY)))


# In[8]:


from sklearn.metrics import precision_score

print('Precision on test data is %.2f' % (precision_score(predY, testY)))


# In[9]:


from sklearn.metrics import recall_score

print('Recall on test data is %.2f' % (recall_score(testY, predY)))


# In[10]:


import numpy as np
from sklearn.metrics import f1_score

print('The F1_score on test data is %.2f' % (f1_score(testY, predY)))


# In[11]:


#4
#Logistic Regression Model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# In[12]:


train_file = "https://raw.githubusercontent.com/binbenliu/Teaching/main/data/Diabetes/diabetes_train.csv"
train_df = pd.read_csv(train_file, header='infer')
train_df


# In[13]:


test_file = "https://raw.githubusercontent.com/binbenliu/Teaching/main/data/Diabetes/diabetes_test.csv"
test_df = pd.read_csv(test_file, header='infer')
test_df


# In[14]:


cols = train_df.columns
cols


# In[15]:


x_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']


# In[16]:


# train data
X_train = train_df[x_cols].values
y_train = train_df['Outcome'].values

# test data
X_test = test_df[x_cols].values
y_test = test_df['Outcome'].values


# In[17]:


import sklearn.linear_model
clf_lr = sklearn.linear_model.LogisticRegression(fit_intercept=True)


# In[18]:


clf_lr.fit(X_train, y_train)


# In[19]:


y_pred_lr = clf_lr.predict(X_test)


# In[20]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


print('Accuracy on test data is %.2f' % (accuracy_score(y_test, y_pred_lr)))
print('Precision on test data is %.2f' % precision_score(y_test, y_pred_lr) )
print('Recall on test data is %.2f' % recall_score(y_test, y_pred_lr) )
print('F1_score on test data is %.2f' % f1_score(y_test, y_pred_lr) )


# In[21]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred_lr, labels=clf_lr.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=clf_lr.classes_)
disp.plot()


# In[22]:


import matplotlib.pyplot as plt


coefs = pd.DataFrame(
   clf_lr.coef_.ravel(),
   columns=['Coefficients'], index=x_cols
)

coefs.plot(kind='barh', figsize=(10, 9), color='green')
plt.title('Logistic regression model')
plt.axvline(x=0, color='0.5')
plt.subplots_adjust(left=.3)
plt.grid( color='0.95')

# From this model, I find that the Outcome feature is the most important, followed by Age and Pregnancies. Age is on the negative side while Pregnancies are on the positive side.


# In[ ]:




