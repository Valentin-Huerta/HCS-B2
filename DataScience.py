# Necessary imports:
import json
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import numpy as np
import random
import os
from flask import Flask, request

# Setting up current directory:
CURRENT_DIRECTORY = os.getcwd()

# Alternatively, use random datasets
rands1 = [random.randint(0,100) for i in range(100)]
rands2 = [random.randint(0,100) for i in range(100)]
df = pd.DataFrame({
    'x': rands1,
    'y': rands2
})

# Using the sklearn KMeans algorithm to group data
kmeans = KMeans(n_clusters=12)
kmeans.fit(df)

# Accessing the central points of each group
centroids = kmeans.cluster_centers_

# Accessing the labels of each point
labels = kmeans.labels_

# Plotting the results below
# fig = plt.figure(figsize=(5, 5))
# ax = fig.add_subplot(1,1,1)
# ax.scatter(df['x'], df['y'], c=labels.astype(np.float), edgecolor='k', s=40)

# Importing data from a csv file
dataset = pd.read_csv(CURRENT_DIRECTORY + '/Temp_and_rain.csv')

# Ways to visualize the data:
rows, cols = dataset.shape

# Check how many of each species we have
dataset.groupby('Month').size()

# splitting up the labels and the values for each species:
feature_columns = ['Temperature']
X = dataset[feature_columns].values
Y = dataset['Month'].values


# Encoding Labels (Turning string species names into integers)
# setosa -> 0
# versicolor -> 1
# virginica -> 2
le = LabelEncoder()
Y = le.fit_transform(Y)

# plt.figure(figsize=(15,10))
# parallel_coordinates("Month")
# plt.title('Parallel Coordinates Plot', fontsize=20, fontweight='bold')
# plt.xlabel('Features', fontsize=15)
# plt.ylabel('Features values', fontsize=15)
# plt.legend(loc=1, prop={'size': 15}, frameon=True,shadow=True, facecolor="white", edgecolor="black")
# plt.show()

# Splitting into training and test datasets:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5, random_state = 0)

# Creating the learning model
knn_classifier = KNeighborsClassifier(n_neighbors=10)

# Fitting the model with the training data
knn_classifier.fit(X_train, Y_train)

# Making predictions with the test data (This line is also where we would potentially classify new data)
Y_pred = knn_classifier.predict(X_test)
print(Y_pred)

# Finding Accuracy:
accuracy = accuracy_score(Y_test, Y_pred)*100
print('Accuracy of model: ' + str(round(accuracy, 2)) + ' %.')
cm = confusion_matrix(Y_test, Y_pred)
print(cm)

# creating list of cv scores
cv_scores = []

# perform 10-fold cross validation
for k in range(1, 50, 2):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())


# Displaying results visually
# plt.figure()
# plt.figure(figsize=(15,10))
# plt.title('The optimal number of neighbors', fontsize=20, fontweight='bold')
# plt.xlabel('Number of Neighbors K', fontsize=15)
# plt.ylabel('Accuracy', fontsize=15)
# plt.plot(cv_scores)
# plt.show()

# Set up Flask App
app = Flask(__name__)

@app.route("/", methods = ['GET'])
def classify():
    # array mapping numbers to flower names
    classes = [ "1","2","3","4","5","6","7","8","9","10","11","12"]

    # get values for each component, return error message if not a float
    try:
        values = [[float(request.args.get(component)) for component in ["Temperature"]]]
    except TypeError:
        return "An error occured\nUsage: 127.0.0.1:5000?Temperature=NUM"

    # Otherwise, return the prediction.
    prediction = knn_classifier.predict(values)[0]

    if(classes[prediction] == "1"):
        return "January"
    elif(classes[prediction] == "2"):
        return "February"
    elif(classes[prediction] == "3"):
        return "March"
    elif(classes[prediction] == "4"):
        return "April"
    elif(classes[prediction] == "5"):
        return "May"
    elif(classes[prediction] == "6"):
        return "June"
    elif(classes[prediction] == "7"):
        return "July"
    elif(classes[prediction] == "8"):
        return "August"
    elif(classes[prediction] == "9"):
        return "Semptember"
    elif(classes[prediction] == "10"):
        return "October"
    elif(classes[prediction] == "11"):
        return "November"
    elif(classes[prediction] == "12"):
        return "Dicember"

# Run the app.
app.run()

# try http://127.0.0.1:5000/?Temperature=16.976
