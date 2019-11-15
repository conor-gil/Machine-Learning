import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('tcd-ml-1920-group-income-train.csv')
dataset.reset_index(drop=True, inplace=True)

dataset.columns = dataset.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('[','').str.replace(']','')

for col in dataset.columns:
    dataset.loc[dataset[col] == -1, col] = np.nan

X_cols = ['year_of_record','housing_situation','crime_level_in_the_city_of_employement','work_experience_in_current_job_years',
              'satisfation_with_employer','gender','age','country','size_of_city',
              'profession','university_degree','wears_glasses','hair_color','body_height_cm',
              'yearly_income_in_addition_to_salary_e.g._rental_income']

y_col = 'total_yearly_income_eur'

#print(dataset.columns)
data = dataset[X_cols + [y_col]]
#print(data)

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

data.dropna(subset=X_cols, inplace=True)

#Need to add 'Yearly Income in addition' and 'Work Experience'

X_cols_numeric = list(data._get_numeric_data().columns)
#print(X_cols_numeric)
X_cols_OHE = ['university_degree','satisfation_with_employer','housing_situation','gender']

#print(X_cols_numeric)
#print(X_cols_OHE)
cols = X_cols_numeric + X_cols_OHE
#print(cols)

X = data[cols]
y = data.total_yearly_income_eur

X = pd.get_dummies(X, columns=['university_degree'])
X = pd.get_dummies(X, columns=['satisfation_with_employer'])
X = pd.get_dummies(X, columns=['housing_situation'])
X = pd.get_dummies(X, columns=['gender'])

#print(X.columns)

city_col = X.size_of_city
city_col2 = np.where(city_col < 3000, 1, 0)
X.size_of_city = city_col2

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2, random_state = 42)

mdl = LinearRegression()
mdl = mdl.fit(xTrain, yTrain)

from sklearn.metrics import mean_squared_error, mean_absolute_error
#mse = mean_squared_error(yTest, mdl.predict(xTest))
#print('Root Mean squared error =', np.sqrt(mse))

yPred = mdl.predict(xTest)

mse = mean_squared_error(yTest, yPred)
print('Root Mean squared error =', np.sqrt(mse))
print('Mean Absolute error =', mean_absolute_error(yTest, yPred))

'''
mdl2 = LinearRegression()
mdl2 = mdl2.fit(X,y)
yPred2 = mdl2.predict(X)
mse2 = mean_squared_error(y, yPred2)
print('Root Mean squared error 2 =', np.sqrt(mse2))

print(y)
print(yPred2)
'''
