import pandas as pd
from pre_processing import preprocess_data, preprocess_input_text
from model import train_and_evaluate_model, load_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import numpy as np

### Info

###  Here below is the code to check the model accuracy if you want and results as we are deploying the ML model 
###  You can add another path when trying to re-run or train the model when running the code so you could uncomment the
###  code below in e.g save_model_path="/app/model.pkl" and for save_vectorizer_path="/app/vect.pkl" in the pre-processing.py file
##   also we have saved the train model and deploying it so it don't have
###  to train everytime we try to predict and hit the api. As it was said in instruction not to upload data on git repo so
###  you would have to put the sample_data.csv in the directory when clone repo so you can read the data in pd.read_csv()
###  as below if you want to run the code below to see the results:

### Test
### You can run the api tests locally when clone the repo just go to the directory and run pytest to run the test

### Result
### The results score were 86% accuracy wise for prediction of labels against the text 

# save_model_path = "put here the path you want to save the train model"

# df = pd.read_csv("/app/sample_data.csv")
# X_train, X_test, y_train, y_test, vectorizer, label_encoder = preprocess_data(df,num_rows=None)

# model, accuracy, report = train_and_evaluate_model(X_train, y_train, X_test, y_test,save_model_path)

# # Display results
# print(f"Accuracy: {accuracy}")
# print("Classification Report:")
# print(report)


labels=['ft', 'mr', 'ct','pkg', 'ch','cnc']
newlabels = np.array(labels)


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from pre_processing import preprocess_input_text

app = FastAPI()

class Item(BaseModel):
    text: str

# Load the trained model and vectorizer
model = joblib.load("/app/model.pkl")
vectorizer = joblib.load("/app/vect.pkl")


@app.post("/predict")
async def predict(item: Item):
    # Preprocess input text
    preprocessed_data = [preprocess_input_text(text) for text in [item.text]]

    new_data_vectorized = vectorizer.transform(preprocessed_data)

    predictions = model.predict(new_data_vectorized)

    predicted_label_text = [newlabels[idx] for idx in predictions]

    return {"predicted_label": predicted_label_text, "input_text": item.text}

# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)