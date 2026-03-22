from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# dataset
X, y = load_iris(return_X_y=True)

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# save model
import os
pickle_path = os.path.join(os.getcwd(),"model.pkl")
pickle.dump(model, open(pickle_path,"wb"))
print(f"model.pkl created at {pickle_path}")