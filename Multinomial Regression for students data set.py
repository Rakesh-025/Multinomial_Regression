### Multinomial Regression ####
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("E:/Data Science 18012022/Multinomial Regression/mdata.csv")
#removing unwanted columns
df = df.drop('Unnamed: 0', axis = 1)
df = df.drop('id', axis = 1)
df.describe()
df.columns
# Converting Categorical Columns into Numeric Columns
df = pd.get_dummies(df, columns = ["female","schtyp","honors"], drop_first = True)
from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
labelencoder = LabelEncoder()
df['ses']= labelencoder.fit_transform(df['ses'])
df.columns

df = df[['prog', 'ses','read', 'write','math','science','female_male','schtyp_public','honors_not enrolled']] # rearranging columns

df.prog.value_counts() # each category count in the variable "prog"

# Boxplot of independent variable distribution for each category of prog 
sns.boxplot(x = "prog", y = "ses", data = df)
sns.boxplot(x = "prog", y = "read", data = df)
sns.boxplot(x = "prog", y = "write", data = df)
sns.boxplot(x = "prog", y = "math", data = df)
sns.boxplot(x = "prog", y = "science", data = df)
sns.boxplot(x = "prog", y = "female_male", data = df)
sns.boxplot(x = "prog", y = "schtyp_public", data = df)
sns.boxplot(x = "prog", y = "honors_not enrolled", data = df)


# Scatter plot for each categorical programe of student
sns.stripplot(x = "prog", y = "ses", jitter = True, data = df)
sns.stripplot(x = "prog", y = "read", jitter = True, data = df)
sns.stripplot(x = "prog", y = "write", jitter = True, data = df)
sns.stripplot(x = "prog", y = "math", jitter = True, data = df)
sns.stripplot(x = "prog", y = "science", jitter = True, data = df)
sns.stripplot(x = "prog", y = "female_male", jitter = True, data = df)
sns.stripplot(x = "prog", y = "schtyp_public", jitter = True, data = df)
sns.stripplot(x = "prog", y = "honors_not enrolled", jitter = True, data = df)


# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 
sns.pairplot(df) # Normal
sns.pairplot(df, hue = "prog") # With showing the category of each prog in the scatter plot

# Correlation values between each independent features
df.corr()

train, test = train_test_split(df, test_size = 0.2)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(train.iloc[:, 1:], train.iloc[:, 0])
help(LogisticRegression)

test_predict = model.predict(test.iloc[:, 1:]) # Test predictions

# Test accuracy 
accuracy_score(test.iloc[:,0], test_predict) #0.6

train_predict = model.predict(train.iloc[:, 1:]) # Train predictions 
# Train accuracy 
accuracy_score(train.iloc[:,0], train_predict)#0.6375

# Test accuracy and Train accuracy is almost same so, we can accept this model, it is a good model we can say