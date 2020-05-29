# Ensemble Model for Kaggle Competition on Beijing Pollution Data
The competition rules where to make a regression model to predict the response variable for an initially unknown dataset. There were over 200 participating teams in this private competition held for University of Toronto students. I participated alone and my model achieved first place on the final leaderboard.

I built a weighted ensemble model consisting of a Linear Generalised Additive Model (GAM) and an Extreme Gradient Boosted Tree Regressor Model (XGBM). 

I went through multiple iterations of training with a plathora of different models before reaching my final model. The details about my entire process can be found on the pdf contained in this repository along with the python code for the final model.

Some of the feature engineering and hyperparameter tuning techniques I applied are Feature Creation, Feature Selection, Bayesian Optimizations and Grid Searches.

## Results

Loss Curve for the model is shown below:
![](https://github.com/saihiel/Ensemble_Regression_Kaggle/blob/master/Learning_curves.png)
![](https://github.com/saihiel/Ensemble_Regression_Kaggle/blob/master/New%20Feature%20Importance.png)
The features found to be the most informative (highest predictive power) were used in the final model.
