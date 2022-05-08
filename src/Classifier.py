
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import  LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import itertools
import sklearn.metrics as metrics


class Classifier(object):

    def __init__(self, countV, test_news, train_news, tfidf_ngram, model_file):
        self.countV = countV
        self.test_news = test_news
        self.train_news = train_news
        self.tfidf_ngram = tfidf_ngram
        self.model_file = model_file

     #the feature selection has been done in FeatureSelection.py module. here we will create models using those features for prediction
     #first we will use bag of words techniques
    def buildClassifier(self):
        #building classifier using naive bayes 
        self.nb_pipeline = Pipeline([
                ('NBCV', self.countV),
                ('nb_clf',MultinomialNB())])

        self.nb_pipeline.fit(self.train_news['Statement'],self.train_news['Label'])
        self.predicted_nb = self.nb_pipeline.predict(self.test_news['Statement'])
        np.mean(self.predicted_nb == self.test_news['Label'])


        #building classifier using logistic regression
        self.logR_pipeline = Pipeline([
                ('LogRCV',self.countV),
                ('LogR_clf',LogisticRegression())
                ])

        self.logR_pipeline.fit(self.train_news['Statement'],self.train_news['Label'])
        self.predicted_LogR = self.logR_pipeline.predict(self.test_news['Statement'])
        np.mean(self.predicted_LogR == self.test_news['Label'])


        #building Linear SVM classfier
        self.svm_pipeline = Pipeline([
                ('svmCV',self.countV),
                ('svm_clf',svm.LinearSVC())
                ])

        self.svm_pipeline.fit(self.train_news['Statement'],self.train_news['Label'])
        self.predicted_svm = self.svm_pipeline.predict(self.test_news['Statement'])
        np.mean(self.predicted_svm == self.test_news['Label'])


        #using SVM Stochastic Gradient Descent on hinge loss
        self.sgd_pipeline = Pipeline([
                ('svm2CV',self.countV),
                ('svm2_clf',SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5))
                ])

        self.sgd_pipeline.fit(self.train_news['Statement'],self.train_news['Label'])
        self.predicted_sgd = self.sgd_pipeline.predict(self.test_news['Statement'])
        np.mean(self.predicted_sgd == self.test_news['Label'])


        #random forest
        self.random_forest = Pipeline([
                ('rfCV',self.countV),
                ('rf_clf',RandomForestClassifier(n_estimators=200,n_jobs=3))
                ])
    
        self.random_forest.fit(self.train_news['Statement'],self.train_news['Label'])
        self.predicted_rf = self.random_forest.predict(self.test_news['Statement'])
        np.mean(self.predicted_rf == self.test_news['Label'])


    ##Now using n-grams
    def buildClassifierUsingNgrams(self):
        #naive-bayes classifier
        self.nb_pipeline_ngram = Pipeline([
                ('nb_tfidf',self.tfidf_ngram),
                ('nb_clf',MultinomialNB())])

        self.nb_pipeline_ngram.fit(self.train_news['Statement'],self.train_news['Label'])
        self.predicted_nb_ngram = self.nb_pipeline_ngram.predict(self.test_news['Statement'])
        np.mean(self.predicted_nb_ngram == self.test_news['Label'])


        #logistic regression classifier
        self.logR_pipeline_ngram = Pipeline([
                ('LogR_tfidf',self.tfidf_ngram),
                ('LogR_clf',LogisticRegression(penalty="l2",C=1))
                ])

        self.logR_pipeline_ngram.fit(self.train_news['Statement'],self.train_news['Label'])
        self.predicted_LogR_ngram = self.logR_pipeline_ngram.predict(self.test_news['Statement'])
        np.mean(self.predicted_LogR_ngram == self.test_news['Label'])


        #linear SVM classifier
        self.svm_pipeline_ngram = Pipeline([
                ('svm_tfidf',self.tfidf_ngram),
                ('svm_clf',svm.LinearSVC())
                ])

        self.svm_pipeline_ngram.fit(self.train_news['Statement'],self.train_news['Label'])
        self.predicted_svm_ngram = self.svm_pipeline_ngram.predict(self.test_news['Statement'])
        np.mean(self.predicted_svm_ngram == self.test_news['Label'])


        #sgd classifier
        self.sgd_pipeline_ngram = Pipeline([
                 ('sgd_tfidf',self.tfidf_ngram),
                 ('sgd_clf',SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5))
                 ])

        self.sgd_pipeline_ngram.fit(self.train_news['Statement'],self.train_news['Label'])
        self.predicted_sgd_ngram = self.sgd_pipeline_ngram.predict(self.test_news['Statement'])
        np.mean(self.predicted_sgd_ngram == self.test_news['Label'])


        #random forest classifier
        self.random_forest_ngram = Pipeline([
                ('rf_tfidf',self.tfidf_ngram),
                ('rf_clf',RandomForestClassifier(n_estimators=300,n_jobs=3))
                ])
    
        self.random_forest_ngram.fit(self.train_news['Statement'],self.train_news['Label'])
        self.predicted_rf_ngram = self.random_forest_ngram.predict(self.test_news['Statement'])
        np.mean(self.predicted_rf_ngram == self.test_news['Label'])

        self.test_news['Label'].shape




    """Out of all the models fitted, we would take 2 best performing model. we would call them candidate models
    from the confusion matrix, we can see that random forest and logistic regression are best performing 
    in terms of precision and recall (take a look into false positive and true negative counts which appeares
    to be low compared to rest of the models)"""

    def calculateClassifierParam(self):  
        #grid-search parameter optimization
        #random forest classifier parameters
        self.parameters = {'rf_tfidf__ngram_range': [(1, 1), (1, 2),(1,3),(1,4),(1,5)],
                       'rf_tfidf__use_idf': (True, False),
                       'rf_clf__max_depth': (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
        }

        self.gs_clf = GridSearchCV(self.random_forest_ngram, self.parameters, n_jobs=-1)
        self.gs_clf = self.gs_clf.fit(self.train_news['Statement'][:10000],self.train_news['Label'][:10000])

        self.gs_clf.best_score_
        self.gs_clf.best_params_
        self.gs_clf.cv_results_

        #logistic regression parameters
        self.parameters = {'LogR_tfidf__ngram_range': [(1, 1), (1, 2),(1,3),(1,4),(1,5)],
                       'LogR_tfidf__use_idf': (True, False),
                       'LogR_tfidf__smooth_idf': (True, False)
        }

        self.gs_clf = GridSearchCV(self.logR_pipeline_ngram, self.parameters, n_jobs=-1)
        self.gs_clf = self.gs_clf.fit(self.train_news['Statement'][:10000],self.train_news['Label'][:10000])

        self.gs_clf.best_score_
        self.gs_clf.best_params_
        self.gs_clf.cv_results_

        #Linear SVM 
        self.parameters = {'svm_tfidf__ngram_range': [(1, 1), (1, 2),(1,3),(1,4),(1,5)],
                       'svm_tfidf__use_idf': (True, False),
                       'svm_tfidf__smooth_idf': (True, False),
                       'svm_clf__penalty': ('l1','l2'),
        }

        self.gs_clf = GridSearchCV(self.svm_pipeline_ngram, self.parameters, n_jobs=-1)
        self.gs_clf = self.gs_clf.fit(self.train_news['Statement'][:10000],self.train_news['Label'][:10000])

        self.gs_clf.best_score_
        self.gs_clf.best_params_
        self.gs_clf.cv_results_

        

        
    #by running above commands we can find the model with best performing parameters
    def findBestPerformingModel(self):

        #running both random forest and logistic regression models again with best parameter found with GridSearch method
        self.random_forest_final = Pipeline([
                ('rf_tfidf',TfidfVectorizer(stop_words='english',ngram_range=(1,3),use_idf=True,smooth_idf=True)),
                ('rf_clf',RandomForestClassifier(n_estimators=300,n_jobs=3,max_depth=10))
                ])
    
        self.random_forest_final.fit(self.train_news['Statement'],self.train_news['Label'])
        self.predicted_rf_final = self.random_forest_final.predict(self.test_news['Statement'])
        np.mean(self.predicted_rf_final == self.test_news['Label'])
        print(metrics.classification_report(self.test_news['Label'], self.predicted_rf_final))

        self.logR_pipeline_final = Pipeline([
                #('LogRCV',countV_ngram),
                ('LogR_tfidf',TfidfVectorizer(stop_words='english',ngram_range=(1,5),use_idf=True,smooth_idf=False)),
                ('LogR_clf',LogisticRegression(penalty="l2",C=1))
                ])

        self.logR_pipeline_final.fit(self.train_news['Statement'],self.train_news['Label'])
        self.predicted_LogR_final = self.logR_pipeline_final.predict(self.test_news['Statement'])
        np.mean(self.predicted_LogR_final == self.test_news['Label'])
        #accuracy = 0.62
        print(metrics.classification_report(self.test_news['Label'], self.predicted_LogR_final))

        """ by running both random forest and logistic regression with GridSearch's best parameter estimation, we found that for random 
        forest model with n-gram has better accuracty than with the parameter estimated. The logistic regression model with best parameter 
        has almost similar performance as n-gram model so logistic regression will be out choice of model for prediction.
        """

        #saving best model to the disk
        pickle.dump(self.logR_pipeline_ngram,open(self.model_file,'wb'))



    #User defined functon for K-Fold cross validatoin
    def build_confusion_matrix(self, classifier):
    
        k_fold = KFold(n_splits=5)
        scores = []
        confusion = np.array([[0,0],[0,0]])

        for train_ind, test_ind in k_fold.split(self.train_news):
            train_text = self.train_news.iloc[train_ind]['Statement'] 
            train_y = self.train_news.iloc[train_ind]['Label']
    
            test_text = self.train_news.iloc[test_ind]['Statement']
            test_y = self.train_news.iloc[test_ind]['Label']
        
            classifier.fit(train_text,train_y)
            predictions = classifier.predict(test_text)
        
            confusion += confusion_matrix(test_y,predictions)
            score = f1_score(test_y,predictions)
            scores.append(score)
    
        return (print('Total statements classified:', len(self.train_news)),
        print('Score:', sum(scores)/len(scores)),
        print('score length', len(scores)),
        print('Confusion matrix:'),
        print(confusion))
    

    #Plotting learing curve
    def plot_learing_curve(self, pipeline,title):
        size = 2550
        cv = KFold(size, shuffle=True)
    
        X = self.train_news["Statement"]
        y = self.train_news["Label"]
    
        pl = pipeline
        pl.fit(X,y)
    
        train_sizes, train_scores, test_scores = learning_curve(pl, X, y, n_jobs=-1, cv=cv, train_sizes=np.linspace(.1, 1.0, 5), verbose=0)
       
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
     
        plt.figure()
        plt.title(title)
        plt.legend(loc="best")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.gca().invert_yaxis()
    
        # box-like grid
        plt.grid()
    
        # plot the std deviation as a transparent range at each training set size
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
        # plot the average training and test score lines at each training set size
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
        # sizes the window for readability and displays the plot
        # shows error from 0 to 1.1
        plt.ylim(-.1,1.1)
        plt.show()


  
    """
    by plotting the learning cureve for logistic regression, it can be seen that cross-validation score is stagnating throughout and it 
    is unable to learn from data. Also we see that there are high errors that indicates model is simple and we may want to increase the
    model complexity.
    """


    #plotting Precision-Recall curve
    def plot_PR_curve(self, classifier):
    
        precision, recall, thresholds = precision_recall_curve(self.test_news['Label'], classifier)
        average_precision = average_precision_score(self.test_news['Label'], classifier)
    
        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,
                         color='b')
    
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Random Forest Precision-Recall curve: AP={0:0.2f}'.format(
                  average_precision))


    #User defined functon for K-Fold cross validatoin
    def build_confusion_matrix(self, classifier, title):
    
        k_fold = KFold(n_splits=5)
        scores = []
        confusion = np.array([[0,0],[0,0]])

        for train_ind, test_ind in k_fold.split(self.train_news):
            train_text = self.train_news.iloc[train_ind]['Statement'] 
            train_y = self.train_news.iloc[train_ind]['Label']
    
            test_text = self.train_news.iloc[test_ind]['Statement']
            test_y = self.train_news.iloc[test_ind]['Label']
        
            classifier.fit(train_text,train_y)
            predictions = classifier.predict(test_text)
        
            confusion += confusion_matrix(test_y,predictions)
            score = f1_score(test_y,predictions)
            scores.append(score)
    
        self.plot_confusion_matrix(confusion, classes=['FAKE Data', 'REAL Data'], title=title)

        return (print('Total statements classified:', len(self.train_news)),
        print('Score:', sum(scores)/len(scores)),
        print('score length', len(scores)),
        print('Confusion matrix:'),
        print(confusion))

    def plot_confusion_matrix(self, cm, classes, title,
                              normalize=False,
                              cmap=plt.cm.Blues):
    
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title('Confusion Matrix ' + title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    
 
    """
    Now let's extract the most informative feature from ifidf vectorizer for all fo the classifiers and see of there are any common
    words that we can identify i.e. are these most informative feature acorss the classifiers are same? we will create a function that 
    will extract top 50 features.
    """

    def show_most_informative_features(self, model, vect, clf, text=None, n=50):
        # Extract the vectorizer and the classifier from the pipeline
        vectorizer = model.named_steps[vect]
        classifier = model.named_steps[clf]

         # Check to make sure that we can perform this computation
        if not hasattr(classifier, 'coef_'):
            raise TypeError(
                "Cannot compute most informative features on {}.".format(
                    classifier.__class__.__name__
                )
            )
            
        if text is not None:
            # Compute the coefficients for the text
            tvec = model.transform([text]).toarray()
        else:
            # Otherwise simply use the coefficients
            tvec = classifier.coef_

        # Zip the feature names with the coefs and sort
        coefs = sorted(
            zip(tvec[0], vectorizer.get_feature_names()),
            reverse=True
        )
    
        # Get the top n and bottom n coef, name pairs
        topn  = zip(coefs[:n], coefs[:-(n+1):-1])

        # Create the output string to return
        output = []

        # If text, add the predicted value to the output.
        if text is not None:
            output.append("\"{}\"".format(text))
            output.append(
                "Classified as: {}".format(model.predict([text]))
            )
            output.append("")

        # Create two columns with most negative and most positive features.
        for (cp, fnp), (cn, fnn) in topn:
            output.append(
                "{:0.4f}{: >15}    {:0.4f}{: >15}".format(
                    cp, fnp, cn, fnn
                )
            )
        #return "\n".join(output)
        print(output)

