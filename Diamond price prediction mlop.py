#!/usr/bin/env python
# coding: utf-8

# # Diamond Price Prediction

# # Problem Statement

# The objective of this project is to develop a predictive model that accurately estimates the price of diamonds based on their various features. The dataset contains information about diamonds, including their carat, cut, color, clarity, and dimensions.
# The goal is to build a machine learning model that can take these features as input and predict the price of diamonds with a high level of accuracy. This model will assist *jewelers, buyers, and sellers* in determining the appropriate price range for diamonds based on their characteristics, enabling them to make informed decisions during transactions.
# 
# The project will involve data preprocessing, exploratory data analysis, feature engineering, and the development and evaluation of predictive models. The model's performance will be assessed using appropriate evaluation metrics, such as root mean squared error or mean absolute percentage error, to measure the accuracy of the price predictions.
# 
# The successful completion of this project will provide a valuable tool for the diamond industry, facilitating pricing decisions and improving transparency in the market.

# In[1]:


#Importing relevant libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns


# In[2]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[3]:


from sklearn import metrics
import statsmodels.api as sm
import plotly.express as px #for plotting the scatter plot
#For plotting the dataset in seaborn
sns.set(style='whitegrid')
import warnings #to remove the warnings
warnings.filterwarnings('ignore')


# In[4]:


#uploading the data set

df = pd.read_csv("Diamond.csv")


# In[5]:


df


# # Data Description

# Content and meaning of each column in the data is as below:
# 
# price - diamond price in US dollars and target variable (continuous)
# 
# carat - weight of the diamond (continuous)
# 
# cut - quality of cut (categorical)
# 
# color - diamond color (categorical)
# 
# clarity - measure of flawless or how clear the diamond is (categorical)
# 
# depth - total depth percentage estimated by formulas (continuous)
# 
# table - width of top of the diamond relative to widest point (continuous)
# 
# x - lenghth in mm (continuous)
# 
# y - width in mm (continuous)
# 
# z - depth in mm (continuous)
# 

# # Exploratory Data Analysis

# In[6]:


#checking first five rows of the data set

df.head()


# In[7]:


#checking last 5 rows of the data set

df.tail()


# In[8]:


#checking the shape

df.shape


# There are total 53,940 rows and 10 columns in the data set

# In[9]:


#checking information about the data

df.info()


# In[10]:


#getting data's description

df.describe()


# In[11]:


#all the columns present in the data set

df.columns


# In[12]:


#all data types

df.dtypes


# In[13]:


#Checking for duplicates

df.loc[df.duplicated(keep='first')]


# We can see 146 duplicate rows

# In[14]:


#Removing duplicate rows

df = df.drop_duplicates()


# In[15]:


df.shape


# We can see we have dropped 146 rows

# In[16]:


#dividing data into categorical and numerical

X_num = df.select_dtypes(include = np.number)
X_cat = df.select_dtypes(exclude = np.number)


# In[17]:


#checking counts of the values

X_cat.value_counts


# # Outliars

# In[18]:


#checking for outliars

df.hist(figsize=(20,10), bins = 30, edgecolor = 'black')
plt.show()


# In[19]:


# filter the numeric variables from the data

df_num = df.select_dtypes(include = np.number)


# In[20]:


# plot the boxplot for each variable
# subplots(): plot subplots
# figsize(): set the figure size

fig, ax = plt.subplots(2, 3, figsize=(15, 8))

# plot the boxplot using boxplot() from seaborn
# z: let the variable z define the boxplot
# x: data for which the boxplot is to be plotted
# orient: "h" specifies horizontal boxplot (for vertical boxplots use "v")
# whis: proportion of the IQR past the low and high quartiles to extend the plot whiskers
# ax: specifies the axes object to draw the plot onto
# set_xlabel(): set the x-axis label
# fontsize: sets the font size of the x-axis label


for variable, subplot in zip(df_num.columns, ax.flatten()):
    z = sns.boxplot(x = df_num[variable], orient = "h",whis=1.5 , ax=subplot) # plot the boxplot
    z.set_xlabel(variable, fontsize = 20)                                     # set the x-axis label


# We can see alot of outliars in multiple columns

# In[21]:


df.mean()


# In[22]:


df.median()


# We can see price column has outliars.

# In[23]:


#Removing outliars

#removing outliers with IQR 

q1 = np.quantile(df.price, 0.25) #1st quantile
q2 = np.quantile(df.price, 0.5) #2nd quantile
q3 = np.quantile(df.price, 0.75) #3rd quantile
IQR = q3 - q1 #Inter - Quartile Range


# In[24]:


IQR


# In[25]:


#setting the fences on our dataset to identify the outliers

lower = q1 - (1.5*IQR) #lower whisker
upper = q3 + (1.5*IQR) #upper whisker


# In[26]:


#removal of lower and upper outliers in our data

df = df[~((df.price > upper) | (df.price < lower))]


# In[27]:


df.shape


# We now have 50271 rows and 10 columns

# # Missing values

# In[28]:


#to see if any columns has empty cells

for column in df:
    print(" Number of empty cells in {} is {} ". format(column, (df[column]=="").sum()))


# In[29]:


#finding the number of zeros in each column

for column in df:
    print(("Number of zeros in {} is {}".format(column,(df[column]==0).sum())))


# From the above table also we see that x (length), y(breadth) and z(depth) have zero as these minimum value. So we need to remove these entries with 0s using the code below. Good news is that there are no empty cells.

# In[30]:


#dropping all zero values from x,y and z columns
df = df[(df[['x','y','z']] != 0).all(axis=1)]


# In[31]:


#to check the minimum values of x,y and z
df.describe()


# In[32]:


df.shape


# We have dropped 11 rows

# # Encoding categorical variables using Ordinal Encoding

# As we know, in our dataset, the columns cut, clarity and colour have non numerical entries. These features or predictors are called as categorical variables as they put the variable entries in various categories. Like in our example, predictor cut has various categories like Ideal, Premium and Good. To deal with categorical entries, techniques like one hot encoding, dummy coding, label encoding etc can be used. In this implementation, built in methods are not used. Instead, the categories in the categorical variables are replaced by the numeric value as per the category significance. This is shown in the code below-

# In[33]:


#Replacing the categorical value colour, cut and clarity without using built in function for categorical data
df=df.replace({'color' : { 'D' : 6, 'E' : 5, 'F' : 4, 'G' : 3, 'H': 2, 'I':1, 'J':0}})
df=df.replace({'cut': {'Ideal':4, 'Premium': 3, 'Very Good':2, 'Good':1, 'Fair':0}})
df=df.replace({'clarity': {"IF": 8, 'VVS1' :7, 'VVS2': 6, 'VS1': 5, 'VS2': 4, 'SI1':3, 'SI2': 2, 'I1':1, 'I2':0, 'I3':0}})
#Visualize the data frame
df.head()


# # Multicollinearity

# Checking for multicollinearity among the independent variables and also identifying non contributing independent variable
# Before we proceed ahead with the regression, it is very important to check for existence of multicollinearity among the independent variable. That ascertaining whether the change in one variable brings about the change in another independent variable also. If that is the case, we should remove one of the variables from the predictor variable list to make the model more accurate. Moreover, we also need to check if all the independent variables in consideration are actually contributing to the change in dependent variable. Removing any such non-contributing independent variable can aid in reducing the Mean Square Error (MSE).

# In[34]:


# Create a correlation matrix between every pair of attributes
corr_matrix = df.corr()

# Plot the correlation with seaborn
plt.subplots(figsize = (10, 8))
sns.heatmap(corr_matrix, annot = True)
plt.show()


# In[35]:


numeric_df = df.select_dtypes(include=[np.number])
numeric_df = sm.add_constant(numeric_df)

from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
vif = [vif(numeric_df, i) for i in range(numeric_df.shape[1])]

pd.DataFrame(vif, index=numeric_df.columns, columns=['vif'])


# In[36]:


def calculate_vif(df):
    # Select only the numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df = sm.add_constant(numeric_df)
    # Calculate VIF values
    vif = np.zeros(numeric_df.shape[1])
    for i in range(numeric_df.shape[1]):
        # extracting the label in loop
        y = numeric_df.iloc[:, i]
        # extracting the all features which is not label
        x = numeric_df.iloc[:, np.arange(numeric_df.shape[1]) != i]
        # fit the  model
        model = LinearRegression().fit(x, y)
        # claculate the R_2
        r_squared = model.score(x, y)
        # calculating VIF
        vif[i] = 1 / (1 - r_squared)

    # Create a DataFrame to store the VIF values
    vif_df = pd.DataFrame({
        "features": numeric_df.columns,
        "VIF": vif
    })

    return vif_df


# In[37]:


calculate_vif(df)


# Now, we know that variable, x, y and z are length, height and depth respective. Product of these variables can give one single variable “volume”. We can use the following code to implement the same.
# Now, we know that variable, x, y and z are length, height and depth respective. Product of these variables can give one single variable “volume”. We can use the following code to implement the same.

# In[38]:


#Reducing three variables x, y, z to a single variable
df['volume']= df['x']*df['y']*df["z"] 
#now we can drop x,y,z columns
df=df.drop(['x','y','z'], axis=1)
#Visualizing the data frame to see the change
df.head()


# In[39]:


#Plot heat map to see the correlation among the variables
corr = df.corr()
plt.figure(figsize = (15,8)) #To set the figure size
sns.heatmap(data=corr, square=True , annot=True, cbar=True)


# In[40]:


def calculate_vif(df):
    # Select only the numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df = sm.add_constant(numeric_df)
    # Calculate VIF values
    vif = np.zeros(numeric_df.shape[1])
    for i in range(numeric_df.shape[1]):
        # extracting the label in loop
        y = numeric_df.iloc[:, i]
        # extracting the all features which is not label
        x = numeric_df.iloc[:, np.arange(numeric_df.shape[1]) != i]
        # fit the  model
        model = LinearRegression().fit(x, y)
        # claculate the R_2
        r_squared = model.score(x, y)
        # calculating VIF
        vif[i] = 1 / (1 - r_squared)

    # Create a DataFrame to store the VIF values
    vif_df = pd.DataFrame({
        "features": numeric_df.columns,
        "VIF": vif
    })

    return vif_df


# In[41]:


calculate_vif(df)


# Since both Carat and volume has high VIF, we'll check for feature significance.

# In[42]:


#Define the independent and dependent variables
y= df['price'] #dependent variable is price
x= df.drop(['price'], axis=1)


# splitting the data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2)


#This is the intercept that has to be added to create the model
x = sm.add_constant(x)

# create the model
modelNew = sm.OLS(y, x)

#fit the model
fitted = modelNew.fit() 

#Obtain the results of regression
fitted.summary()


# Carat and Volume both are physical features of a Diamond. We can see Carat is more significant feature as it has t value of 53.561 and volume has 12.023 as t value. Thus, we are dropping volume.

# In[43]:


df = df.drop('volume', axis=1)


# In[44]:


df.head() 


# In[45]:


# plotting heatmap
def plot_heatmap():
    from seaborn import heatmap
    
    # define correlation matrix
    corr_df = df.corr(method='pearson')
    df_lt = corr_df.where(np.tril(np.ones(corr_df.shape)).astype(np.bool))
    
    # plot heatmap
    plt.figure(figsize=(8, 6))
    heatmap(data=df_lt, annot=True, square=True, cbar=True, linewidths=.5, fmt='.3f', cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

plot_heatmap()


# # Scaling the data

# In[46]:


#Built-in function to standardize the data
from sklearn.preprocessing import StandardScaler
cols=['carat', 'depth','table'] #identifying the columns to be standardized
for i in cols:
#fit the training data with standard scale
    scale = StandardScaler().fit(df[[i]])
# standardize the numerical predictor columns in the dataframe
    df[i] = scale.transform(df[[i]])


# In[47]:


df.head()


# In[48]:


df.value_counts()


# # Linear Regression using Stats Model

# In[49]:


#Define the independent and dependent variables
y= df['price'] #dependent variable is price
x= df.drop(['price'], axis=1)


# In[50]:


# splitting the data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2)


# In[51]:


#This is the intercept that has to be added to create the model
x = sm.add_constant(x)

# create the model
modelNew = sm.OLS(y, x)

#fit the model
fitted = modelNew.fit() 

#Obtain the results of regression
fitted.summary() 


# # Observation

# We can see :
# 
# the P value of our F statistic, and conclude that our overall model is significant.
# 
# The P valus of our t statistic of every feature, and conclude that all our features are significant.

# # Decision tree

# In[58]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error


# In[60]:


# Create a decision tree regressor model and fit the training data
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)

# Predict the target variable for the test data
y_pred = dt.predict(x_test)

# Evaluate the model performance using mean squared error and R-squared score



MAPE_DT = mean_absolute_percentage_error(y_test, y_pred)


print("Mean Absolute Percentage Error (MAPE):", MAPE_DT)


# In[61]:


def calculate_adjusted_r_squared(y_true, y_pred, n_features):
    n = len(y_true)
    residuals = y_true - y_pred
    rss = np.sum(residuals**2)
    tss = np.sum((y_true - np.mean(y_true))**2)
    r_squared = 1 - (rss / tss)
    adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - n_features - 1)
    return adjusted_r_squared

n_features = x_test.shape[1]  # number of features in the test data
adj_r2 = calculate_adjusted_r_squared(y_test, y_pred, n_features)
print("Adjusted R-squared value is", adj_r2)


# In[62]:


n_features = x_test.shape[1]  # number of features in the test data
adj_r2 = calculate_adjusted_r_squared(y_test, y_pred, n_features)
print("Adjusted R-squared value is", adj_r2)


# # Observations

# In[ ]:



print("MAPE for LR is", MAPE_LR)

print("MAPE for KNN is", MAPE_KNN)

print("MAPE for Decision Tree is", MAPE_DT)

print("MAPE for random forest is", MAPE_RF)

print("We see that Random forest gives us the least MAPE")


# In[ ]:


print("RMSE for LR is", RMSE_LR)

print("RMSE for KNN is", RMSE_KNN)

print("RMSE for Decision Tree is", RMSE_DT)

print("RMSE for random forest is", RMSE_RF)

print("We see that Random forest gives us the least RMSE")


# In[ ]:


Metrics_df = pd.DataFrame({
    "Model": ["Linear Regression","KNN",'Decision Tree',"Random Forest"],
    "RMSE": [RMSE_LR, RMSE_KNN, RMSE_DT, RMSE_RF],
    "MAPE": [MAPE_LR, MAPE_KNN, MAPE_DT, MAPE_RF]
})


# In[72]:


Metrics_df


# In[ ]:





# In[73]:


# Pickle


# In[63]:


import pickle


# In[64]:


# Assuming you have X_train and y_train as your training features and target variable
model = DecisionTreeRegressor()
model.fit(x_train, y_train)


# In[65]:


# Store the trained model in a variable
model = model


# In[66]:


# Predict on training set
train_predictions = model.predict(x_train)

# Predict on test set
test_predictions = model.predict(x_test)


# In[67]:



def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

train_mape = mean_absolute_percentage_error(y_train, train_predictions)
test_mape = mean_absolute_percentage_error(y_test, test_predictions)


# In[68]:


print("Train MAPE:", train_mape)
print("Test MAPE:", test_mape)


# In[69]:


import pickle


# In[70]:


# Assuming your trained model is stored in a variable called 'model'
with open('gb.pkl', 'wb') as file:
    pickle.dump(model, file)


# In[93]:


import gzip


# In[94]:


with gzip.open("gb.pklz","wb") as file:
    pickle.dump(model, file)


# In[72]:


import streamlit as st


# In[74]:


def preprocess_input(carat, cut, color, clarity, depth, table):
    # Create a dataframe with the input data
    data = pd.DataFrame({'carat': [carat], 'cut': [cut], 'color': [color], 'clarity': [clarity],
                         'depth': [depth], 'table': [table]})
    
    # Perform any necessary encoding or preprocessing on the data
    
    # Example: One-hot encoding for categorical features
    data_encoded = pd.get_dummies(data, columns=['cut', 'color', 'clarity'])
    
    # Get the column names of the encoded data
    feature_names = data_encoded.columns.tolist()
    
    # Convert the encoded data to a numpy array
    data_array = data_encoded.values
    
    return data_array, feature_names

# Define the Streamlit app
def main():
    st.title("Diamond Price Prediction")

    # Add the necessary input fields for diamond characteristics
    carat = st.number_input("Carat")
    cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
    color = st.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])
    clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])
    depth = st.number_input("Depth")
    table = st.number_input("Table")

    # Preprocess the input data
    new_data, feature_names = preprocess_input(carat, cut, color, clarity, depth, table)

    # Make predictions using the loaded model
    predictions = model.predict(new_data)

    # Display the predictions
    st.subheader("Predicted Price:")
    st.write(predictions[0])

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[79]:


# Store the trained model in a variable
model = model


# In[ ]:





# In[80]:


# Assuming 'X_test' contains the test features and 'y_test' contains the corresponding target variable
predictions = model.predict(x_test)


# In[81]:


from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, predictions)


# In[82]:


mse


# In[83]:



def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_test, predictions)


# In[84]:


mape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[85]:


# Pickle the model and write it to a file
with open('diamond_price_model.pickle', 'wb') as file:
    pickle.dump(model, file)


# In[ ]:




