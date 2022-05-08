# Fake News Detection
In this project, we have used various natural language processing techniques and machine learning algorithms to classify fake news articles using sci-kit libraries from python.

## Prerequisites

-Python 3.6</br>
-Sklearn (scikit-learn)</br>
-numpy</br>
-scipy</br>
 
The dataset used in this project is in the .csv format, the original .tsv files can be found in liar_dataset.
## Steps to run
Step 1: Clone this github repository and set it as your working directory by the following command:
```
!git clone https://github.com/dharace/Fake-News-Detection.git
!cd /content/Fake-News-Detection
```
Step 2: Install all the dependencies from the Requirements.txt
```
pip install -r Requirements.txt
```
Step 3: Import required classes from github to call methods.</br>
Step 4: Install dataset.</br>
Step 5: Go to the "data" folder & download glove.6B.zip.</br>
Step 6: Create object of DataPrep.py and pass filepath of each dataset test.csv, train.csv, valid.csv.</br>
Step 7: Create object of FeatureSelection.py and pass filepath of glove_6B_50d.</br>
Step 8: Create object of Classifier.py and pass CountVectorizer object, test & train dataset, tfidf_ngram and final model.</br>
Step 9: Test the result.</br>
Preview of the code can be accessed through this [ipynb](https://colab.research.google.com/github/dharace/Fake-News-Detection/blob/main/TestFakeNewsDetection.ipynb#scrollTo=Dc3QFmjhCfF6) notebook.
At the end of the program, you will be asked for an input which will be a piece of information or a news headline that you want to verify. Once you paste or type news headline, then press enter.
The output of the true or false news will be produced along with the probability.
## Baseline Model

|               | Precision | Recall | F1 score |
|---------------|-----------|--------|----------|
| [AENeT](https://link.springer.com/article/10.1007/s00521-021-06450-4)         | 0.63      | 0.57   | 0.60     |
| [Deep Ensemble](https://arxiv.org/abs/1811.04670) | 0.55      | 0.45   | 0.43     |
