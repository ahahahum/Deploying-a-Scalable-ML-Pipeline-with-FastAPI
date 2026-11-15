# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model was created by Muhaiminul Islam in November 2026 as part of a Udacity course project on deploying a scalable machine learning pipeline. It is a random forest classifier from scikit-learn (version 1.5.1) with 100 trees and a fixed random seed of 42. All other hyperparameters are the scikit-learn defaults, and no hyperparameter tuning was performed. The categorical features are one-hot encoded and the salary label is binarized. The trained model, encoder, and label binarizer are saved as pickle files in the 'model/' folder and are served through a FastAPI application. 

## Intended Use

The model predicts whether a person's annual income exceeds $50,000 based on demographic and employment attributes from US census data. It was built to demonstrate a complete machine learning workflow: training, evaluating on data slices, testing, continuous integration, and deployment through a REST API. It is intended for educational use. It should not be used to make real decisions about individuals, such as decisions about credit, employment, housing, or insurance.

## Training Data

The model was trained on the UCI Census Income (Adult) dataset, which contains 32,561 rows and 15 columns: 14 features and the 'salary' target. The features are age, workclass, fnlgt, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, and native-country, The target is 'salary', which is either '<=50K' or '>50K'. About 75.9% of the rows are '<=50K' and 24.1% are '>50K', so the classes are imbalanced. Missing values are marked with '?' in the workclass (1,836 rows), occupation (1,843 rows), and native-country (583 rows) columns. These were kept and treated as their own category. The data was split into 80% for training (20,048 rows) and 20% for testing (6,513 rows) using a stratified split, so both sets keep the same class balance. During preprocessing, eight categorical features (workclass, education, marital-status, occupation, relationship, race, sex, and native-country) were one-hot encoded, the continuous features were used as they were, and the label was binarized. The encoder and binarizer were fitted on the training data only. 

## Evaluation Data

The model was evaluated on the 20% held-out test set of 6,513 rows. It was never used for fitting the model, the encoder, or the label binarizer. It was processed with the encoder and binarizer learned from the training data. The same test set was used to compute the overall metrics and the metrics on data slices.

## Metrics
_Please include the metrics used and your model's performance on those metrics._

The model was evaluated with precision, recall, and the F1 score (F-beta with beta equal to 1). Precision is the share of people predicted to earn more than $50,000 and recall is the share of people who earn more than $50,000 that the model finds. F1 is the harmonic mean of the two. Accuracay alone was not used, because a model that always predicted '<=50K' would be correct about 76% of the time. On the test set, the model achieved a precision of 0.7353, a recall of 0.6368, and a F1 score of 0.6831. This means the model is correct about 74% of the time when it predicts a high income and it finds about 64% of the high earners so it misses more high earners than it falsely flags. The performance was also computed on slices of the test data, one for each value of each categorical feature. The full results are in 'slice_output.txt'. Performance varies across groups. For example, for sex, the model scored a precision of 0.7680, a recall of 0.6082, and a F1 of 0.6788 for women (2,158 test rows) and a precision of 0.7298, a recall of 0.6432, and a F1 of 0.6838 for for men (4,355 test rows). The F1 scores are close but the model is more precise and less sensitive for women, which means it misses a larger share of women who earn more than $50,000. For race the F1 score was 0.6861 for White people (5,533 rows) and 0.6750 for Black people (662 rows), while the scores for smaller groups were lower or less stable: 0.6263 for Asian-Pac-Islander (200 rows) and 0.5333 for American-Indian-Eskimo (73 rows). The Other group reached an F1 of 0.8571, but it has only 45 rows so that score should not be trusted. Slices with very few rows, such as some native-country and workclass values, produce unreliable scores including perfect 1.0000 or zero values because a handful of examples can change the result a lot.

## Ethical Considerations

The data contain sensitive attributes including race, sex, and native country and it reflects historical patterns of income inequality in the United States. A model trained on it can learn and repeat those patterns, so its predictions may be less accurate or less fair for some groups. The slice results show that performance is not the same for every group which is a sign of this. The model also uses attributes such as marital status and relationship that can act as proxies for other characteristics. Using it to make decisions about real people could cause harm so it should only be used for learning purposes. The census data was collected from individuals and although it is public and anonymized, any use of it should respect the privacy of the people it describes.

## Caveats and Recommendations

The data is from the 1994 census and covers only the United States so the model may not reflect current incomes or other countries. The classes are imbalanced which lowers recall of the '>50K' group, and the model was not tuned to correct for this. Some slices contain too few examples to be trusted. The 'fnlgt' feature is a sampling weight rather tha na personal attribute and the '?' values were kept as a category instead of being imputed. For future work, I recommend tuning the hyperparameters, trying class weighting, or other methods for imbalanced data, comparing other algorithms, checking fairness metrics across groups, and retraining on more recent data before any further use. 