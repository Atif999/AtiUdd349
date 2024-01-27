from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.svm import SVC

def train_and_evaluate_model(X_train, y_train, X_test, y_test,save_model_path=None):
    
    clf=SVC() 
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    if save_model_path:
        joblib.dump(clf, save_model_path)
        print(f"Model saved at: {save_model_path}")
    
   
    return clf, accuracy, report

def load_model(model_path):

    model = joblib.load(model_path)
    return model