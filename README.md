# *Recipe Site Traffic Prediction*

Using machine learning techniques, predict high-traffic recipes on a recipe website.

These predictions will assist the website's product manager in making data-driven decisions to enhance user engagement and overall traffic on the site.

## Business goals

Provide the prediction for the recipes that will lead to high traffic.

Provide the high traffic recipes with at least 80% accuracy.

### Methods Used:
* Data Cleaning
* Data Visualization
* Predictive Modeling
* Model Evaluation
* Business Metrics
 
### Technologies
* Python / Jupyter Notebook
* NumPy, Pandas, Seaborn, Matplotlib, sklearn

# Dataset information:

![github](https://github.com/Khangtran94/Recipe_High_Traffic_Prediction/assets/146164801/dfaa923a-2408-4ed6-8655-1659b0546ce0)

![github](https://github.com/Khangtran94/Recipe_High_Traffic_Prediction/assets/146164801/ea254f66-7a9d-436c-978d-0b5820193bb5)

## Data Validation:

![github](https://github.com/Khangtran94/Recipe_High_Traffic_Prediction/assets/146164801/63067fbc-359e-404b-af6f-7a867fea1ad3)

## Exploratory data analysis (EDA):
![github](https://github.com/Khangtran94/Recipe_High_Traffic_Prediction/assets/146164801/2e04aa9d-1fb7-4c3a-a5ba-5ba138b74abc)

According to the graph above, there are 535 recipes labeled as high traffic account for 59.78%, while there are 360 recipes labeled as low traffic taking up 40.22%. ==> The high-traffic recipes are higher than the low-traffic recipes.

![github](https://github.com/Khangtran94/Recipe_High_Traffic_Prediction/assets/146164801/863e2ff4-9c74-457c-a930-b3d5c3db27e0)

The graphs above illustrate the distribution of servings and categories in the dataset.

* Servings: The majority of recipes (approximately 60%) are served for four people, while the number of recipes served for one, two, and six people are almost the same.
* Categories: The most common category in the dataset is chicken, accounting for a significant portion of the recipes.

![github](https://github.com/Khangtran94/Recipe_High_Traffic_Prediction/assets/146164801/48688cd9-2397-4629-88d7-e2f2b9d298cb)

Two graphs above show us the relationship between the target variables - traffic and two categorical columns: servings and category.
* Traffic vs Servings: For each of servings, the amount of recipes with high traffic is higher than the amount of recipes with low traffic. No significant correlation between servings and high_traffic is seen.
* Traffic vs Category: Except for three categories (Chicken, Breakfast and Beverages), seven categories remain always have the high-traffic recipes higher than the low-traffic recipes. Potato, Vegetable, Pork are the highest in high traffic recipes. The category can influence high-traffic.

![github](https://github.com/Khangtran94/Recipe_High_Traffic_Prediction/assets/146164801/718dfcb3-85ac-480d-8a94-d7a7153da56a)

There are no major differences in nutrient distribution between the recipe with high-traffic and the recipe with low-traffic.

![github](https://github.com/Khangtran94/Recipe_High_Traffic_Prediction/assets/146164801/4e8225f3-6d4d-47ba-9e61-267a1d03d286)

* The category that contains the highest level of calories is Chicken then Pork and One Dish Meal.
* The category that contains the highest level of carbohydrate is Potato.
* The category that contains the highest level of sugar is Dessert.
* The category that contains the highest level of protein is Chicken then One Dish Meal, Pork, Lunch/Snacks and Meat.

## Model Development: 
This is a binary classification problem. I choose ***LogisticRegression*** and ***LinearSVC***

![github](https://github.com/Khangtran94/Recipe_High_Traffic_Prediction/assets/146164801/13d0c8e8-8b8d-4ce8-9aa5-fcb75dc6c7bb)

## Model Evaluation:

![github](https://github.com/Khangtran94/Recipe_High_Traffic_Prediction/assets/146164801/68603654-e569-429c-a3a1-ff1935ad5d84)

* Both the Logistic Regression model and LinearSVC model worked well with the dataset.
* Both models let us know that the category is the most effective feature for the high traffic.
* According to the results from the above steps, the Logistic Regression satisfied both business goals. Its precision, recall and F1-score have values equal to or greater than 80%.

## Business Metrics:

![github](https://github.com/Khangtran94/Recipe_High_Traffic_Prediction/assets/146164801/b8542450-d1dc-43e9-b07a-3fae2737e7b8)

* The Logistic Regression model achieved an accuracy of 78%
* While the Linear SVC model had a slightly lower accuracy of 76%.

# Recommendations for future action:
* We suggest to deploy the Logistic Regression model to the recent recipes. With approximately 78% in predict high-traffic recipes, this predictive model can assist the product manager reaches the business goals in generating more traffic to the websit and boost overall performances.
* Both models suggest that category is the main feature affecting the traffic. Therefore, we should try to increase the number of categories and create more meaningful features from existing variables.
* To improve the accuracy, we should collect more information, such as more details about time to cost, cost per servings, and also the combination of ingredients.
