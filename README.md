# Python-Red-wine-Analysis
Analysis of Factors Affecting Red Wine Quality | Python

## Introduction
The preliminary investigation found that residual sugar, all acidity and alcohol content together constitute the balance of wine flavor. The acidity provides the depth of flavor to the wine and at the same time ensures the refreshing taste of the wine after neutralization with the residual sugar, while the alcohol content adds more burning and sharp sensation to the taste. The quality of red wine seems to depend mainly on three factors: residual sugar, acidity and alcohol.
Based on the above information, the business question of this capstone project is to understand the following through analysis:
1. The quality of red wine mainly depends on which factors.
2. Whether there is any connection between the various factors that affect the quality of red wine.

## Data Set
There are two files in the original data set, with red wine and white wine respectively. In this project, only the red wine data set would be analyzed and conducted, namely winequality-white.csv. The red-wine dataset contains 12 columns with 19,176 records in total. The 12 variables include wine quality, fixed acidity, volatile acidity, citric acid, residual sugar, density, pH, alcohol and so on. In the process of analysis, the quality of red wine will be used as the dependent variable, and the other 11 characteristic variables will be used to predict the quality of red wine.

## Modeling
In the modeling stage, we tried three models to explore the variables that can affect the quality of red wine. By comparing the multiple logistic regression model and the decision tree classifier model, the results showed that neither of these models had high prediction accuracy. We then chose the random forest model with the highest accuracy, which has a high accuracy in predicting wine quality, while observing the weights that affect wine quality. Among them, alcohol level is the most important influencing characteristic. 

## Conclusion
From the red wine quality distribution chart we can see that most of the red wine quality is between 5-7, and no red wine quality can reach 9. The first two models, logistic regression model and decision tree model established in the experiment have reached the results of predicting the data features of the data set with more than 50% accuracy. However, to get a better fit model for the prediction, we used a random forest model, re-dividing the data then obtained an accuracy at 91.5% which would be our final model to capture the wine quality factors.  From the random forest model’s feature importance graph we can see that the most influential variable is the alcohol at about 16% followed by volatile acidity at about 12%. Therefore, our recommendation for red wine merchants is to pay more attention to alcohol, volatile acidity, sulphates in red wine, so as to produce high-quality red wine.
