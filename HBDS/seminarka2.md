# Seminar Task n.2 - Pavel Falta

## Model Selection

In this seminar project, I was tasked with trying out different machine learning models to find the best ones for predicting student performance based on various features. I also had to implement bayesian hyper-parameter search and stacking model.

## Preprocessing

Before training the models, I applied the following preprocessing steps:

First, I removed the columns `G1`, `G2`, because of the strong correlation and used `G3` as the target variable. For the categorical features, I applied label encoding to convert them into numerical values. For the numerical features, I used standard scaling to normalize the data.

To categorize the target variable `G3` into `n_groups` equal groups, I calculated the quantiles and created bins accordingly. This allowed me to transform the continuous target variable into discrete categories. I found that splitting the data into just two groups worked the best, as we didn't have a lot of data to begin with.

## Ensemble Technique

To boost the performance of the models, I used ensemble techniques as taught in class. Specifically, I used:

1. **Stacking**: This method combines multiple classifiers via a meta-classifier. The base classifiers are trained on the original dataset, and the meta-classifier is trained on the outputs of the base classifiers.

## Performance Comparison

### Training and Evaluation

We used a 70/30 split for training/testing.

Each model was trained on the training dataset (`X_train`, `y_train`) and then tested on the test dataset (`X_test`, `y_test`). I measured performance using Accuracy and the Confusion Matrix.

### Results

- **Stacking Model**: The stacking model, which combined Random Forest, SVC, KNN, and XGBoost as base classifiers with a Decision Tree as the meta-classifier, showed the highest accuracy and provided a balanced performance across different metrics. On the maths dataset, the model achieved an accuracy of around 63-68%, while on the Portuguese dataset, it achieved an accuracy of around 70-75%. With 3 groups, the model performed slighty worse on both datasets and hovered around ~65%.

### Visualization

To make the results clearer, I visualized them by plotting the confusion matrix for the stacking model.

## Conclusion

In summary, the stacking model proved to be a great tool in my toolbox, which both ensures realiable accuracy and solid results for predicting student performance. Using ensemble techniques like stacking made the model predictions more reliable and improved overall performance by a significant amount.

