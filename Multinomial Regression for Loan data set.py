### Multinomial Regression ####
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("E:/Data Science 18012022/Multinomial Regression/loan.csv")
#removing unwanted columns
df = df.drop(['id','member_id','term','grade','sub_grade','emp_title','emp_length','home_ownership','verification_status','issue_d','pymnt_plan','url','desc','purpose','title','zip_code','addr_state','earliest_cr_line','application_type','initial_list_status','int_rate'], axis = 1)
# Dropping the columns having NaN/NaT values
df = df.dropna(axis=1)
df.describe()
df.columns
df.info()

df = df[['loan_status','loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'installment','annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths','open_acc', 'pub_rec', 'revol_bal', 'total_acc', 'out_prncp','out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp','total_rec_int', 'total_rec_late_fee', 'recoveries','collection_recovery_fee', 'last_pymnt_amnt', 'policy_code','acc_now_delinq', 'delinq_amnt']] # rearranging columns

# Boxplot of independent variable distribution for each category of loan_status 

sns.boxplot(x = "loan_status", y = "loan_amnt", data = df)
sns.boxplot(x = "loan_status", y = "funded_amnt", data = df)
sns.boxplot(x = "loan_status", y = "funded_amnt_inv", data = df)
sns.boxplot(x = "loan_status", y = "installment", data = df)
sns.boxplot(x = "loan_status", y = "annual_inc", data = df)
sns.boxplot(x = "loan_status", y = "dti", data = df)
sns.boxplot(x = "loan_status", y = "delinq_2yrs", data = df)
sns.boxplot(x = "loan_status", y = "inq_last_6mths", data = df)
sns.boxplot(x = "loan_status", y = "open_acc", data = df)
sns.boxplot(x = "loan_status", y = "pub_rec", data = df)
sns.boxplot(x = "loan_status", y = "revol_bal", data = df)
sns.boxplot(x = "loan_status", y = "total_acc", data = df)
sns.boxplot(x = "loan_status", y = "out_prncp", data = df)
sns.boxplot(x = "loan_status", y = "out_prncp_inv", data = df)
sns.boxplot(x = "loan_status", y = "total_pymnt", data = df)
sns.boxplot(x = "loan_status", y = "total_pymnt_inv", data = df)
sns.boxplot(x = "loan_status", y = "total_rec_prncp", data = df)
sns.boxplot(x = "loan_status", y = "total_rec_int", data = df)
sns.boxplot(x = "loan_status", y = "total_rec_late_fee", data = df)
sns.boxplot(x = "loan_status", y = "recoveries", data = df)
sns.boxplot(x = "loan_status", y = "collection_recovery_fee", data = df)
sns.boxplot(x = "loan_status", y = "last_pymnt_amnt", data = df)
sns.boxplot(x = "loan_status", y = "policy_code", data = df)
sns.boxplot(x = "loan_status", y = "acc_now_delinq", data = df)
sns.boxplot(x = "loan_status", y = "delinq_amnt", data = df)


# Scatter plot for each categorical programe of student
sns.stripplot(x = "loan_status", y = "loan_amnt", jitter = True, data = df)
sns.stripplot(x = "loan_status", y = "funded_amnt", jitter = True, data = df)
sns.stripplot(x = "loan_status", y = "funded_amnt_inv", jitter = True, data = df)
sns.stripplot(x = "loan_status", y = "installment", jitter = True, data = df)
sns.stripplot(x = "loan_status", y = "annual_inc", jitter = True, data = df)
sns.stripplot(x = "loan_status", y = "dti", jitter = True, data = df)
sns.stripplot(x = "loan_status", y = "delinq_2yrs", jitter = True, data = df)
sns.stripplot(x = "loan_status", y = "inq_last_6mths", jitter = True, data = df)
sns.stripplot(x = "loan_status", y = "open_acc", jitter = True, data = df)
sns.stripplot(x = "loan_status", y = "pub_rec", jitter = True, data = df)
sns.stripplot(x = "loan_status", y = "revol_bal", jitter = True, data = df)
sns.stripplot(x = "loan_status", y = "total_acc", jitter = True, data = df)
sns.stripplot(x = "loan_status", y = "out_prncp", jitter = True, data = df)
sns.stripplot(x = "loan_status", y = "out_prncp_inv", jitter = True, data = df)
sns.stripplot(x = "loan_status", y = "total_pymnt", jitter = True, data = df)
sns.stripplot(x = "loan_status", y = "total_pymnt_inv", jitter = True, data = df)
sns.stripplot(x = "loan_status", y = "total_rec_prncp", jitter = True, data = df)
sns.stripplot(x = "loan_status", y = "total_rec_int", jitter = True, data = df)
sns.stripplot(x = "loan_status", y = "total_rec_late_fee", jitter = True, data = df)
sns.stripplot(x = "loan_status", y = "recoveries", jitter = True, data = df)
sns.stripplot(x = "loan_status", y = "collection_recovery_fee", jitter = True, data = df)
sns.stripplot(x = "loan_status", y = "last_pymnt_amnt", jitter = True, data = df)
sns.stripplot(x = "loan_status", y = "policy_code", jitter = True, data = df)
sns.stripplot(x = "loan_status", y = "acc_now_delinq", jitter = True, data = df)
sns.stripplot(x = "loan_status", y = "delinq_amnt", jitter = True, data = df)


# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 
sns.pairplot(df) # Normal
sns.pairplot(df, hue = "loan_status") # With showing the category of each loan_status in the scatter plot

# Correlation values between each independent features
df.corr()

train, test = train_test_split(df, test_size = 0.2)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(train.iloc[:, 1:], train.iloc[:, 0])
help(LogisticRegression)

test_predict = model.predict(test.iloc[:, 1:]) # Test predictions

# Test accuracy 
accuracy_score(test.iloc[:,0], test_predict) #0.9998741188318228

train_predict = model.predict(train.iloc[:, 1:]) # Train predictions 

# Train accuracy 
accuracy_score(train.iloc[:,0], train_predict)#0.9999685267365374

# Test accuracy and Train accuracy is same so, we can accept this model, it is a good model we can say