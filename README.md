# beginner_ML_project
`python` `pandas` `scikit-learn`\
\
Inspired by [Programming with Mosh](https://www.youtube.com/watch?v=7eh4d6sabA0), this is a beginner data science project utilizing a small, provided dataset provided by Mosh. The decision tree model created will predict what genre of music an individual would like based on their gender and age.

## Importing Data
After importing the data, it is evident that this is small dataset since it only contains 18 records. It is important to analyze the data to see if there are any inconsistencies or null values. To keep this project simple, the data was cleaned ahead of time.
> ```python
> import pandas as pd
>
> dataframe = pd.read_csv('music.csv')
> ```

## Splitting Data
When analyzing the data, we see there is an 'age', 'gender' (male = 1, female = 0), and 'genre' column. Since the goal of this project is predicting the genre for the missing gender and age ranges, the genre is known as the *target* (or label). The *target* column needs isolated from the rest of the data. This is depicted as `y`.\
\
In order for our model to work, *features* need to be identified (depicted as `X`). These *features* will be used to train our model so our model can learn underlying patterns to then predict our *target*. In this scenario, our features are age and gender.
> ```python
> X = dataframe.drop(['genre'], axis=1)
> y = dataframe['genre']
> ```

> [!NOTE]
> Do not include the target attribute within your features

\
The target and features have been identified. The next step is to split these into a training set and a test set. A machine learning standard is to allocate 20% towards the test set, and a `random_state=42`. Setting a random state allows the randomized split to be reproducible which is beneficial when following along. 
> ```python
> from sklearn.model_selection import train_test_split
>
> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
> ```

## Creating and Training the Model
There are many different models that can be utilized in machine learning. For this project, we are trying to _classify_ our _target_, or _label_, based on an individual's age and gender. There are only 5 genres in our dataframe (Classical, Dance, HipHop, Acoustic, and Jazz). Since there are a limited amount of _labels_ (genres), we will utilize a `DecisionTreeClassifier`. An instance of this class needs to be created. Once created, we need to train this model using our training data to avoid data leakage. This is known as <ins>fitting a model</ins>.
> ```python
> from sklearn.tree import DecisionTreeClassifier
>
> model = DecisionTreeClassifier()
> model.fit(X_train, y_train)
> ```

## Accuracy
Now that our model is trained, we want to see how accurate our model's predictions are. The _test features_ are used during the predictions. To test accuracy, the _test labels_ are the expected values which is calculated against the actual values, which are the predictions. Accuracy is scored on a scale of 0-1 with 0 meaning not accurate at all and 1 meaning 100% accuracy.
> ```python
> from sklearn.metrics import accuracy_score
>
> predictions = model.predict(X_test)
>
> acc = accuracy_score(y_test, predictions)
> ```

## Visualizing the Model
To help truly understand how the `DecisionTreeClassifer` model works, you can visualize the results. A decision tree is essentially a flowchart. `plot_tree` can be used to show the tree's branches and nodes based on the features and class names (target values). Keep in mind that only the unique values of the target attribute should be used.
> ```python
> from skearn import tree
>
> tree.plot_tree(model, feature_names=['age','gender'], class_names=sorted(y.unique()))

## Recap
We touched on a simple machine learning project to predict what genre of music a person may like based on their age and gender. After importing the data, the target attribute and features were identified. These were then split into test sets and training sets. A decision tree was instantiated and fitted. We then calculated how accurate our model's predictions were. To gain more knowledge on decision trees, we visualized what the one we created looked like.\
\
I hope you learned something new or enjoyed this project walk-through 🙂
