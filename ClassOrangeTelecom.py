#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
import pandas as pd
from flask import Flask,Request,jsonify


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,recall_score
from sklearn.metrics import roc_auc_score,roc_curve,scorer,precision_score
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[8]:


class OrangeTelecome():
    def __init__(self):
        self.model          = RandomForestClassifier()
        self.is_trained     = False
        #self.le = lable
    
    
    def get_data(self):
        self.df = pd.read_csv('/Users/mariemashraf/Documents/Machinify/OrangeTelecom/Orange_Telecom_Churn_Data.csv')
    
    

    
    def normalize(self):
        self.cols_to_norm = ['state' , 'number_vmail_messages' , 'total_day_minutes' , 'total_day_calls' , 'total_day_charge' ,
                'total_eve_minutes' , 'total_eve_calls', 'total_eve_charge', 'total_night_minutes', 'total_night_calls',
                'total_night_charge', 'total_intl_minutes', 'total_intl_calls',
                'total_intl_charge', 'number_customer_service_calls']
        self.df[self.cols_to_norm] = self.df[self.cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))


        
    def clean(self):
        self.df = self.df.drop(columns=['account_length' , 'phone_number'] , axis=1 )
        self.df[['state' , 'intl_plan' ,'area_code', 'voice_mail_plan' , 'voice_mail_plan' , 'churned']] = self.df[['state' , 'intl_plan' ,'area_code', 'voice_mail_plan' , 'voice_mail_plan' , 'churned']].apply(LabelEncoder().fit_transform)
        self.normalize()

        
        
    def preprocess(self):
        self.get_data()
        self.clean()
        self.x1 = self.df.drop(columns=['churned'] , axis=1 )
        self.y1 = self.df['churned']
            
        
        
    
    def train(self):
        self.preprocess()
        self.x1_train , self.x1_test ,self.y1_train , self.y1_test = train_test_split(self.x1,self.y1,test_size = 0.25 , random_state=42)
        self.model.fit(self.x1_train ,self.y1_train)
        print("Model Is Trained Successfull ")
        
        self.is_trained = True 
    
    
    def evaluate(self):
        if self.is_trained==False:
            self.train()
            
            return round(self.model.score(self.x1_test,self.y1_test)*100,2)
        else:
            y_pred = self.model.predict(self.x1_test)
            print("Accuarcy For Your Model" ,self.model.score(self.x1_test ,self.y1_test )*100)
            #score = cross_val_score(self.model,self.x1,self.y1,cv=5)
            #print("Cross Validation For Your Model" , np.mean(score)*100)
    
            #R2_regression = round(r2_score(self.y1_test, y_pred),4)
            #print("R2 Regression For Your Model" , R2_regression)
            #MAE_regression = int(mean_absolute_error(self.y1_test,y_pred))
            #print("Mean Absolute Error For Your Model" , MAE_regression)
    
            #print("Confusion Matrix" , confusion_matrix(self.y1_test,y_pred))    
            #print('\n clasification report:\n', classification_report(self.y1_test,y_pred))            
    
    def predict(self,test):
        return
    


# In[9]:


Orange = OrangeTelecome()


# In[10]:


Orange.model = RandomForestClassifier()


# In[11]:


Orange.get_data()


# In[12]:


Orange.df.head()


# In[13]:


Orange.clean()


# In[14]:


Orange.df.head()


# In[15]:


Orange.preprocess()


# In[16]:


Orange.df.head()


# In[17]:


Orange.train()


# In[18]:


Orange.evaluate()


# In[21]:


#Orange.predict(0.32,1,0,1,0.480769,0.754196,0.666667,0.754183,0.542755,0.582353,0.542866,0.619494,0.520000,0.619584,0.500,0.15,0.500000,0.111111)


# In[ ]:





# In[ ]:





# In[ ]:


app = Flask(__name__)
Orange = OrangeTelecome()

@app.route('/')
def home():
    return "Hello Home"


@app.route('/train')
def train():
    Orange.train()
    data = {"status_code":200 , "message":"Model Train successfully"}
    return jsonify(data)

#@app.route('/predict')
#def predict():
#    text = request.args['text']
#    data = {"status_code":200 ,"text": text, "accuracy":Orange.predict(text)}
#    return jsonify(data)

@app.route('/evaluate')
def evaluate():
    data = {"status_code":200 , "accuracy":Orange.evaluate()}
    return jsonify(data)
    #res = Orange.evaluate()
    #return ("Model Accuracy Is : ")+str(res)
    
    



@app.route('/predict')
def predict():
    state                             = request.args['state']
    area_code                         = request.args['area_code']
    intl_plan                         = request.args['intl_plan']
    voice_mail_plan                   = request.args['voice_mail_plan']
    number_vmail_messages             = request.args['number_vmail_messages']
    total_day_minutes                 = request.args['total_day_minutes']
    total_day_calls                   = request.args['total_day_calls']
    total_day_charge                  = request.args['total_day_charge']
    total_eve_minutes                 = request.args['total_eve_minutes']
    total_eve_calls                   = request.args['total_eve_calls']
    total_eve_charge                  = request.args['total_eve_charge']
    total_night_minutes               = request.args['total_night_minutes']
    total_night_calls                 = request.args['total_night_calls']
    total_night_charge                = request.args['total_night_charge']
    total_intl_minutes                = request.args['total_intl_minutes']
    total_intl_calls                  = request.args['total_intl_calls']
    total_intl_charge                 = request.args['total_intl_charge']
    number_customer_service_calls     = request.args['number_customer_service_calls']
    test = [ self,state, area_code, intl_plan, voice_mail_plan,number_vmail_messages,total_day_minutes, total_day_calls,total_day_charge, total_eve_minutes,total_eve_calls,total_eve_charge, total_night_minutes, total_night_calls, total_night_charge, total_intl_minutes, total_intl_calls,total_intl_charge,number_customer_service_calls]
    data = {"status_code":200 ,"text": text, "accuracy":Orange.predict(test)}
    return jsonify(data)



app.run()


# In[ ]:





# In[ ]:





# In[ ]:




