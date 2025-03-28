# Seminar Task n.1 - Pavel Falta

## Model Selection

In my seminar project, I was tasked with trying out different machine learning models to find the best ones for two datasets: the Glass dataset and the Breast Cancer dataset.

## Ensemble Technique

To boost the performance of the models, I used ensemble techniques as i was taught in class. Specifically, I used:

1. **Majority Voting**: This method combines the predictions of multiple classifiers and picks the class that gets the most votes.
2. **Majority Voting (Weighted)**: Similar to majority voting, but each classifier's vote is weighted based on its performance.

I applied these methods using a premade `MajorityVoteClassifier` class, which we worked with in class.

## Performance Comparison

### Training and Evaluation

We used 80/20 split for training/verification.

Each model was trained on the training dataset (`X_train`, `y_train`) and then tested on the test dataset (`X_test`, `y_test`). I measured performance using Accuracy, F1 Score, ROC AUC (only on binary datasets, for my case i used it on Breast Cancer) and the Confusion Matrix.

### Results

- **Glass Dataset**: As expected, the Random Forest (RF) model was the best for the Glass dataset, showing higher accuracy and better F1 scores than other models.
- **Breast Cancer Dataset**: A surprise was that for the Breast Cancer dataset, Logistic Regression was the top performer, with the best accuracy and F1 score.

### Visualization

To make the results clearer, I visualized them by plotting confusion matrices and other performance metrics for each model. I also plotted the ROC curve for the breast cancer dataset.

## Conclusion

In summary, the Random Forest model is the best choice for the Glass dataset, while Logistic Regression works best for the Breasts dataset. Using ensemble techniques like Majority Voting and Weighted Majority Voting made the model predictions more reliable. I chose the weights depending on the best models for the current dataset.