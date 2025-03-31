import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt

data=pd.read_csv('ex.csv')
numeric_columns=data.select_dtypes(include=np.number).columns

scaler=StandardScaler()
data_scaled=scaler.fit_transform(data)

#IF
iso=IsolationForest(contamination=0.1, random_state=42)
iso_forest_pred=iso.fit_predict(data_scaled)
data['IF']=iso_forest_pred

#EE
ee=EllipticEnvelope(contamination=0.09, random_state=42)
ee_pred=ee.fit_predict(data_scaled)
data['EE']=ee_pred

#cook's dist
X=sm.add_constant(data_scaled)
y=np.random.rand(len(data))

ols_model=sm.OLS(y,X).fit()
influence=ols_model.get_influence()
cooks_distance=influence.cooks_distance[0]
data['CC']=cooks_distance

#heatmap
out_col=['IF','EE','CC']
out_data=data[out_col]
plt.figure(figsize=(10,6))
sns.heatmap(out_data,cmap='coolwarm',annot=False,cbar=True)
plt.show()


print("OUTLIER FOR IF:",len(data[data['IF']==-1]))
print("OUTLIER FOR IF:",len(data[data['EE']==-1]))
print("top 5 highest cook's dist:",data.nlargest(5,'CC'))
