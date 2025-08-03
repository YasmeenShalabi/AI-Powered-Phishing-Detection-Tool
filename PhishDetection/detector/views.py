from django.shortcuts import render

import pandas as pd #To handle and preprocess dataset

from sklearn.feature_extraction.text import CountVectorizer #Converting text data into numerical data so ML model can process

from sklearn.model_selection import train_test_split #Split data into training and testing sets

from sklearn.naive_bayes import MultinomialNB #To build phish detection classifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix #Measure performance of ML model

from .forms import MessageForm

dataset= pd.read_csv('C:/Users/17325/Downloads/Phishing_Email.csv/Phishing_Email.csv')

#Preprocess the data
#Handle missing values in the 'Email Text' column
dataset['Email Text'] = dataset['Email Text'].fillna('')

#Convert labels to numerical values: 1 for "Phishing Email" and 0 for "Safe Email"
dataset['Label'] = dataset['Email Type'].apply(lambda x: 1 if x == 'Phishing Email' else 0)

#Use Count Vectorizer to convert text data into numerical format that ML can process
vectorizer= CountVectorizer()
X= vectorizer.fit_transform(dataset['Email Text'])
y = dataset['Label']  #Target labels

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2) #Split data into train(80%) and test(20%) sets

#Train the Naive Bayes classifier
model= MultinomialNB()
model.fit(X_train, y_train)

#Function to predict whether or not a message is phishing
def predictMessage(message):
    messageVector= vectorizer.transform([message])
    prediction= model.predict(messageVector)
    return 'Phish' if prediction[0] == 1 else 'Safe'

def Home(request):
    result= None
    if request.method == 'POST':
        form= MessageForm(request.POST)
        if form.is_valid():
            message= form.cleaned_data['text']
            result= predictMessage(message)

    else:
        form= MessageForm()

    return render(request, 'home.html', {'form': form, 'result': result})