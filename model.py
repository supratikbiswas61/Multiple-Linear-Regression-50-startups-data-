import pandas as pd
import matplotlib.pyplot as plt
import pickle
# Data Exploration
df = pd.read_csv('50_Startups.csv')
# As we have a Categorical variable in the dataset 'State' we will encode it using the get dummies method
State_x = df.iloc[:, :-1]
State_x.head()
s = pd.get_dummies(State_x['State'], drop_first=True)
State_x = State_x.drop('State', axis=1)
State_x = pd.concat([State_x, s], axis=1)
df_test = pd.concat([State_x, df['Profit']], axis=1)
df1 = df_test
X = df1[['R&D Spend', 'Administration', 'Marketing Spend', 'Florida', 'New York']]
y = df1['Profit']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
# Saving model to disk
pickle.dump(lm, open('MLR.pkl', 'wb'))
model = pickle.load(open('MLR.pkl', 'rb'))
print(model.predict([[165300,137896,47100,1,0]]))
