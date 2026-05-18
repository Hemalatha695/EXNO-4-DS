# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
```

```
df = pd.read_csv("/content/bmi.csv")
df
```

<img width="359" height="512" alt="560761411-21152e28-ab1a-4413-a3ad-c8c300c1abcf" src="https://github.com/user-attachments/assets/abb59996-a53b-43db-931b-f97c8bdb37b4" />

```
df.head()
```

<img width="321" height="239" alt="560761560-14ac61ca-1f5a-4515-af21-639896e8f230" src="https://github.com/user-attachments/assets/a5785b67-1d50-4a9f-a50a-0f94ba20deb8" />

```
df.dropna()
```

<img width="346" height="502" alt="560761723-eb4bbf05-5c5e-4545-86f3-da7b0cda0876" src="https://github.com/user-attachments/assets/3d8e2e5d-33f7-4c2f-8950-1a5dcc36af85" />

```
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df[['Hs','Ws']] = sc.fit_transform(df[['Height','Weight']])
df.head(10)
```

<img width="507" height="428" alt="560762107-a6e5083d-dd23-45d6-b36d-c81e491186d2" src="https://github.com/user-attachments/assets/5a701ad1-8c16-49e2-a361-52c4c29a8094" />


```
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
df[['Hm','Wm']] = sc.fit_transform(df[['Height','Weight']])
df.head(10)
```

<img width="701" height="433" alt="560762383-db944b3f-00db-413f-be90-d75d2df856b3" src="https://github.com/user-attachments/assets/b872d332-e9ce-484f-83bc-466e3a732074" />

```
import pandas as pd
df = pd.read_csv("/content/titanic_dataset.csv")
df.columns
```

<img width="710" height="72" alt="560762689-c96fbbeb-34b2-45db-9a42-f25e0d42713a" src="https://github.com/user-attachments/assets/c378dd00-16df-448c-821d-dca4933f1a64" />

```
df
```

<img width="1429" height="515" alt="560763106-f49382d5-24aa-4c16-b1f4-dbd820277d49" src="https://github.com/user-attachments/assets/e687948d-621a-418b-8d1f-e142315d5820" />

```
df.isnull().sum()
```

<img width="179" height="561" alt="560763364-e83ba62a-b0f1-4278-9c41-3f1ffdb02d3c" src="https://github.com/user-attachments/assets/6d105450-1b65-4c4c-aedc-8f8beefd1244" />

```
from sklearn.feature_selection import SelectKBest
X = df[['PassengerId','Pclass','Age','SibSp','Parch','Fare']]
y = df['Survived']
from sklearn.feature_selection import f_classif
selector = SelectKBest(score_func=f_classif, k=4)
X_new = selector.fit_transform(X,y)
selected_feature_indices = selector.get_support(indices=True)
selected_features =X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```

<img width="615" height="56" alt="560763823-ac61d7a6-926c-4361-a947-7d8937838268" src="https://github.com/user-attachments/assets/59074318-dead-4970-936b-1794bfec3fd8" />

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```

<img width="533" height="244" alt="560763973-3460c613-50b3-4c6d-abdb-f3eb159b1a75" src="https://github.com/user-attachments/assets/80b07b89-d363-4e9f-adff-b04c411210d1" />


```
tips.time.unique()
```

<img width="444" height="64" alt="560764078-49a6e0b4-56b2-45ef-b6c4-bd6dafee5701" src="https://github.com/user-attachments/assets/8ba75efd-3272-44d4-93f8-caab8805cd93" />


```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```

<img width="229" height="99" alt="560764194-7e240394-5b0c-4fd3-8bca-4782281efbb2" src="https://github.com/user-attachments/assets/ccc0e5e8-f165-4324-9275-102372b5c94b" />

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```

<img width="400" height="62" alt="560764331-42a599ad-7eff-420f-9e8f-9015dd5b6ca4" src="https://github.com/user-attachments/assets/a10ec6ac-074f-423c-8c24-242492ea4921" />

# RESULT:
   Thus, Feature selection and Feature scaling has been used on the given dataset.  
