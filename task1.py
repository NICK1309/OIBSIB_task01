#import libraries 
from sklearn.datasets import load_iris
iris = load_iris()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy) # printing the accuracy 

sample = [[5.6, 3, 4, 1.5],[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)

# The prediction of the sample is done here
print("Prediction:", prediction) 
species_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
predicted_species = species_names[prediction[0]]


# The prediction of species is done here :
print("Predicted species:", predicted_species)
import pickle
with open('Iris.excel', 'wb') as file: # importing dataset Iris.excel 
    pickle.dump(model, file)