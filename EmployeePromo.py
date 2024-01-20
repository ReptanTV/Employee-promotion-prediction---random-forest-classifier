import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , MinMaxScaler , LabelEncoder
from sklearn.impute import SimpleImputer
from scipy import stats
from scipy.stats import boxcox
from scipy.stats import yeojohnson
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
kFold = StratifiedKFold(n_splits=5)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix , accuracy_score
from joblib import dump

#importing dataset
df = pd.read_csv("C:\\Users\\sasta\\OneDrive\\Desktop\\datasets\\employee_promotion.csv",sep=';')
#-----------------------------------------------------------------------------------------------------------------------------------------------
#filling in missing values with median
prv_yr_r = 'previous_year_rating'
imputer = SimpleImputer(strategy='median')
df[prv_yr_r] = imputer.fit_transform(df[[prv_yr_r]])

avg_tr_s = 'avg_training_score'
imputer = SimpleImputer(strategy='median')
df[avg_tr_s] = imputer.fit_transform(df[[avg_tr_s]])

age = 'age'
imputer = SimpleImputer(strategy='median')
df[age] = imputer.fit_transform(df[[age]])

trainings = 'no_of_trainings'
imputer = SimpleImputer(strategy='median')
df[trainings] = imputer.fit_transform(df[[trainings]])
#-----------------------------------------------------------------------------------------------------------------------------------------------
'''
#Converting the categorical datatypes present in dataset into numeric datatypes
label_encoder = LabelEncoder()
df['department'] = label_encoder.fit_transform(df['department'].values)
print("Class to Encoded Value Mapping:")
for class_name, encoded_value in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
    print(f"{class_name}: {encoded_value}")

df['region'] = label_encoder.fit_transform(df['region'].values)
print("Class to encoded value mapping:")
for class_name , encoded_value in zip(label_encoder.classes_,label_encoder.transform(label_encoder.classes_)):
    print(f"{class_name}: {encoded_value}")
    
df['education'] = label_encoder.fit_transform(df['education'].values)
print("Class to encoded value mapping:")
for class_name , encoded_value in zip(label_encoder.classes_,label_encoder.transform(label_encoder.classes_)):
    print(f"{class_name}: {encoded_value}")
    
df['gender'] = label_encoder.fit_transform(df['gender'].values)
print("Class to encoded value mapping:")
for class_name , encoded_value in zip(label_encoder.classes_,label_encoder.transform(label_encoder.classes_)):
    print(f"{class_name}: {encoded_value}")
    
df['recruitment_channel'] = label_encoder.fit_transform(df['recruitment_channel'].values)
print("Class to encoded value mapping:")
for class_name , encoded_value in zip(label_encoder.classes_,label_encoder.transform(label_encoder.classes_)):
    print(f"{class_name}: {encoded_value}")
    '''
#-----------------------------------------------------------------------------------------------------------------------------------------------

#using boxcox or yeojhonson transformation for normalizing the data
'''
transformed_data, lmbda = yeojohnson(df['department'])
df['department'] = transformed_data

z_scores = np.abs(stats.zscore(df['department']))
thr = 3
out_indices = np.where(z_scores > thr)
'''

transformed_data, lmbda = yeojohnson(df['previous_year_rating'])
df['previous_year_rating'] = transformed_data

z_scores = np.abs(stats.zscore(df['previous_year_rating']))
thr = 3
out_indices = np.where(z_scores > thr)

transformed_data, lmbda = yeojohnson(df['avg_training_score'])
df['avg_training_score'] = transformed_data

z_scores = np.abs(stats.zscore(df['avg_training_score']))
thr = 3
out_indices = np.where(z_scores > thr)

transformed_data, lmbda = yeojohnson(df['no_of_trainings'])
df['no_of_trainings'] = transformed_data

z_scores = np.abs(stats.zscore(df['no_of_trainings']))
thr = 3
out_indices = np.where(z_scores > thr)

transformed_data, lmbda = yeojohnson(df['age'])
df['age'] = transformed_data

z_scores = np.abs(stats.zscore(df['age']))
thr = 3
out_indices = np.where(z_scores > thr)
#-----------------------------------------------------------------------------------------------------------------------------------------------

#putting in x  , y values for predictors and target variables
x = df.drop(columns=['is_promoted','region','gender','recruitment_channel','department','education'])
y = df['is_promoted']
#-----------------------------------------------------------------------------------------------------------------------------------------------

x_train ,x_test ,y_train , y_test = train_test_split(x,y , train_size=0.8)
#-----------------------------------------------------------------------------------------------------------------------------------------------

#running random forest classifier
rfc = RandomForestClassifier(n_estimators=500)
rfc.fit(x_train , y_train)
y_pred = rfc.predict(x_test)
dump(rfc, 'EmployeePromo.joblib')

print("confusion matrix for random forest")
print("_______________________________________________________")
cm = confusion_matrix(y_test,y_pred)
print(cm)
print("_______________________________________________________")

acc = accuracy_score(y_test,y_pred)
print("accuracy score for random forrest is: {}".format(acc * 100),"%")



