import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mnso
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

#### Import dataset
df = pd.read_csv('recipe_site_traffic_2212.csv')
df.head()

# Check size of dataset
df.shape
print("The dataset has {} rows and {} columns.".format(df.shape[0],df.shape[1]))

# Overview the dataset
df.info()

df.describe()

# Number of missing values in each columns
df.isnull().sum()

# Number of unique value of categorical columns
num_unique_objects = df.select_dtypes(include='object').nunique()
unique_values = df.select_dtypes(include='object').apply(lambda x: x.unique())
num_unique_objects,unique_values

# Count values in column category
df.category.value_counts()

# Merge category Chicken Breast into Chicken
df['category'] = df['category'].replace('Chicken Breast', 'Chicken')

# Count values in servings category
df.servings.value_counts()

# Get numeric character
df['servings'] = df['servings'].str.extract(r'(\d+)')

# Change servings column to integer type
df['servings'] = df['servings'].astype('int')

# Count values in column high traffic (include null values)
df['high_traffic'].value_counts(dropna=False)

# Replace missing values with "Low"
df['high_traffic'] = df['high_traffic'].fillna("Low")

# Number of missing values
df.isnull().sum()

# Show the distributions of missing values in the dataset
mnso.matrix(df)

mnso.heatmap(df)

# Rows have missing values
missing_values = df[df.isnull().any(axis=1)]
missing_values

# Calculate percent of missing values
percent_missing = (missing_values.isnull().sum() / len(df)) * 100
percent_missing

# Remove rows have missing values
df.dropna(axis=0,inplace=True)
df.shape

# Impute missing values
nutrient = ['calories','carbohydrate','sugar','protein']
for col in nutrient:
    df[col] = df[col].fillna(df.groupby(['category','servings'])[col].transform('mean'))
    
    # Remove column Recipe
df.drop('recipe',axis=1,inplace=True)

df.describe()

# Check number of duplicate values
num_duplicates = df.duplicated().sum()
num_duplicates

for col in nutrient:
    df['total_'+col] = df[col] * df['servings']
    
df.head()

df.info()

# Count plot for high traffic
sns.countplot(data=df,x='high_traffic')
plt.title('Number of Recipe vs Traffic Status')
plt.xlabel("Traffic Status")

# Calculate number of high, low traffic recipes and their percentage, respectively
high, low = df['high_traffic'].value_counts()

print('Number of recipe labeled as high:',high,'recipes and it accounts for {}%.'.format(round(high/len(df)*100,2)))
print('Number of recipe labeled as low:',low,'recipes and it accounts for {}%.'.format(round(low/len(df)*100,2)))

fig, axes = plt.subplots(1,2,figsize=(12,6))

# Countplot for servings
sns.countplot(data=df, x='servings',ax=axes[0])
axes[0].set_title('Number of Recipes by Servings')
axes[0].set_xlabel("Servings")

# Countplot for category (sorted descending)
axes[1].set_title("Number of Recipes by Category")
axes[1].set_xlabel("Category")
axes[1].set_ylabel("Count")
axes[1].tick_params(axis='x', rotation=90)
sns.countplot(data=df, x="category", ax=axes[1], order=df["category"].value_counts().index)

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Countplot for servings for each high-traffic
axes[0].set_title("Number of Recipes by Servings for Each High-Traffic")
axes[0].set_xlabel("Servings")
axes[0].set_ylabel("Count")
sns.countplot(data=df, x="servings", hue="high_traffic", ax=axes[0])

# Countplot for category for high-traffic (sorted descending)
count = df[df['high_traffic'] == 'High']['category'].value_counts().sort_values(ascending=False)
axes[1].set_title("Number of Recipes by Category for High-Traffic")
axes[1].set_xlabel("Count")
axes[1].set_ylabel("Category")
sns.countplot(data=df,hue='high_traffic', y="category", ax=axes[1], order=count.index)
plt.tight_layout()
plt.show()

# Pairplot for total calories, total carbohydrate, total sugar, total protein
sns.pairplot(data=df,vars=['total_calories','total_carbohydrate','total_sugar','total_protein'])

# Heatmap for total calories, total carbohydrate, total sugar, total protein, and servings
plt.figure(figsize=(10, 8))
sns.heatmap(df[["total_calories", "total_carbohydrate", "total_sugar", "total_protein", "servings"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap for Total Nutrients and Servings")
plt.show()

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Boxplot for calories
sns.boxplot(data=df, x="calories", ax=axes[0, 0])
axes[0, 0].set_title("Calories")

# Boxplot for carbohydrate
sns.boxplot(data=df, x="carbohydrate", ax=axes[0, 1])
axes[0, 1].set_title("Carbohydrate")

# Boxplot for sugar
sns.boxplot(data=df, x="sugar", ax=axes[1, 0])
axes[1, 0].set_title("Sugar")

# Boxplot for protein
sns.boxplot(data=df, x="protein", ax=axes[1, 1])
axes[1, 1].set_title("Protein")

# Adjust the layout and create suptitle
plt.tight_layout()
fig.suptitle('Distribution of each nutrient ',y=1.03,fontsize=16)

# Show the plot
plt.show()

# Create a 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Boxplot for total calories
sns.boxplot(x=df['high_traffic'], y=df['total_calories'], ax=axes[0, 0])
axes[0, 0].set_title('Total Calories')
axes[0, 0].set_xlabel('Traffic Status')
axes[0, 0].set_ylabel('Total Calories')

# Boxplot for total carbohydrate
sns.boxplot(x=df['high_traffic'], y=df['total_carbohydrate'], ax=axes[0, 1])
axes[0, 1].set_title('Total Carbohydrate')
axes[0, 1].set_xlabel('Traffic Status')
axes[0, 1].set_ylabel('Total Carbohydrate')

# Boxplot for total sugar
sns.boxplot(x=df['high_traffic'], y=df['total_sugar'], ax=axes[1, 0])
axes[1, 0].set_title('Total Sugar')
axes[1, 0].set_xlabel('Traffic Status')
axes[1, 0].set_ylabel('Total Sugar')

# Boxplot for total protein
sns.boxplot(x=df['high_traffic'], y=df['total_protein'], ax=axes[1, 1])
axes[1, 1].set_title('Total Protein')
axes[1, 1].set_xlabel('Traffic Status')
axes[1, 1].set_ylabel('Total Protein')

# Adjust spacing between subplots and create suptitle
plt.tight_layout()
fig.suptitle('Compare nutrients with each labeled recipes',y=1.03,fontsize=16)

# Show the plot
plt.show()

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Violinplot for calories by category
sns.violinplot(data=df, y="category", x="calories", ax=axes[0, 0])
axes[0, 0].set_title("Calories by Category")

# Violinplot for carbohydrate by category
sns.violinplot(data=df, y="category", x="carbohydrate", ax=axes[0, 1])
axes[0, 1].set_title("Carbohydrate by Category")

# Violinplot for sugar by category
sns.violinplot(data=df, y="category", x="sugar", ax=axes[1, 0])
axes[1, 0].set_title("Sugar by Category")

# Violinplot for protein by category
sns.violinplot(data=df, y="category", x="protein", ax=axes[1, 1])
axes[1, 1].set_title("Protein by Category")

# Adjust the layout and create suptitle
plt.tight_layout()
fig.suptitle('Compare nutrients with each category',y=1.03,fontsize=16)

# Show the plot
plt.show()

# Replace values in the high-traffic column:
df['high_traffic'] = df['high_traffic'].replace({'High':1,'Low':0})

feature_num = nutrient + ['servings']
# Features = X
X = df[feature_num + ['category']]

# Target = y
y = df['high_traffic']

# split the data with 30% test size
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=29)

# Normalize the data
scaler = StandardScaler()
X_train[feature_num] = scaler.fit_transform(X_train[feature_num])
X_test[feature_num] = scaler.transform(X_test[feature_num])

# Label the category column
encode = LabelEncoder()
X_train['category'] = encode.fit_transform(X_train['category'])
X_test['category'] = encode.transform(X_test['category'])

# The category after label encoder
label_category = dict(zip(encode.classes_,range(len(encode.classes_))))
label_category

# Create a Logistic Regression model
lr = LogisticRegression()

# Define hyperparameter grid for Logistic Regression
grid_lr = {'C':np.logspace(-3,3,7),                 
           'penalty':['l1','l2','elasticnet',None],  
           'multi_class':['auto','ovr','multinomial']} 
    
    # C: inverse of regularization strength 
    # penalty: regularization penalty to be used
    # multi_class: strategy for handling multiple classes
    
# Perform GridSearchCV with 10 folds
lr_cv = GridSearchCV(lr, param_grid=grid_lr,cv=10)

# Fit the model with training data
lr_cv.fit(X_train,y_train)

# Print the best parameters:
print('Tuning hyperparameters for Logistic Regression: ',lr_cv.best_params_)

# Create a Logistic Regression model with the hyperparameters founded:
lr_tuning = LogisticRegression(C = 0.001, multi_class = 'multinomial', penalty='l2')

# Fit and predict with the model
lr_tuning.fit(X_train,y_train)
y_pred_lr = lr_tuning.predict(X_test)

# Create a LinearSVC model
svc = LinearSVC()

# Define hyperparameter grid for LinearSVC
grid_svc = {'C':np.logspace(-3,3,7),                 
           'penalty':['l1','l2'],  
           'loss':['hinge','squared_hinge']} 
    
    # C: inverse of regularization strength 
    # penalty: regularization penalty to be used
    # loss: loss function to be used
    
# Perform GridSearchCV with 10 folds
svc_cv = GridSearchCV(svc, param_grid=grid_svc,cv=10)

# Fit the model with training data
svc_cv.fit(X_train,y_train)

# Print the best parameters:
print('Tuning hyperparameters for LinearSVC: ',svc_cv.best_params_)

# Create a LinearSVC model with the hyperparameters founded:
svc_tuning = LinearSVC(C = 0.1, loss = 'hinge', penalty='l2')


# Fit and predict with the model
svc_tuning.fit(X_train,y_train)
y_pred_svc = svc_tuning.predict(X_test)

# Confusion matrix of Logistic Regression:
print('Confusion matrix of Baseline model - Logistic Regression:')
print(confusion_matrix(y_test,y_pred_lr))
print('\n')
# Classification report of Logistic Regression:
print('Classification report of Baseline model - Logistic Regression: ')
print(classification_report(y_test,y_pred_lr))

# Confusion matrix of LinearSVC:
print('Confusion matrix of Comparison model - LinearSVC:')
print(confusion_matrix(y_test,y_pred_svc))
print('\n')
# Classification report of LinearSVC:
print('Classification report of Comparison model - LinearSVC: ')
print(classification_report(y_test,y_pred_svc))

# Get the coefficients and column names
coefficients_lr = lr_tuning.coef_[0]
coefficients_svc = svc_tuning.coef_[0]
columns = X.columns

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Plot the bar chart for Logistic Regression
axs[0].bar(columns, coefficients_lr)
axs[0].set_xlabel('Features')
axs[0].set_ylabel('Coefficients')
axs[0].set_title('Feature Importance in Logistic Regression Model')
axs[0].tick_params(axis='x', rotation=45)

# Plot the bar chart for LinearSVC
axs[1].bar(columns, coefficients_svc)
axs[1].set_xlabel('Features')
axs[1].set_ylabel('Coefficients')
axs[1].set_title('Feature Importance in LinearSVC Model')
axs[1].tick_params(axis='x', rotation=45)

# Display the subplots
plt.tight_layout()
plt.show()

import random
random.seed(24)
random_recipe_index = random.randint(0, len(X_test) - 1)
random_recipe_features = X_test.iloc[random_recipe_index]

# Reshape the features to match the model's input shape
random_recipe_features = random_recipe_features.values.reshape(1, -1)

# Make the prediction using the Logistic Regression model
predicted_traffic = lr_tuning.predict(random_recipe_features)[0]
if predicted_traffic == 1:
    predicted_traffic_category = "High"
else:
    predicted_traffic_category = "Low"

# Get the actual traffic category from the test set
actual_traffic = y_test.iloc[random_recipe_index]
if actual_traffic == 1:
    actual_traffic_category = "High"
else:
    actual_traffic_category = "Low"

# Print the results
print("Predicted Traffic Category: ", predicted_traffic_category)
print("Actual Traffic Category: ", actual_traffic_category)

# Accuracy score of Logistic Regression:
test_accuracy_lr = accuracy_score(y_test,y_pred_lr)

# Accuracy score of LinearSVC:
test_accuracy_svc = accuracy_score(y_test, y_pred_svc)

print("Accuracy score when using Logistic Regression: ", test_accuracy_lr)
print("Accuracy score when using Linear SVC: ", test_accuracy_svc)


# Bar graph
models = ['Logistic Regression', 'LinearSVC']
accuracy_scores = [test_accuracy_lr, test_accuracy_svc]

plt.bar(models, accuracy_scores)
plt.xlabel('Models')
plt.ylabel('Accuracy Score')

# Add value labels to the bars
for i, v in enumerate(accuracy_scores):
    plt.text(i, v, str(round(v, 2)), ha='center', va='bottom')

plt.title('Comparison of Accuracy Scores between Models')
plt.show()