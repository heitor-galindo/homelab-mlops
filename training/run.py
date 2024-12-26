import os
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

# set the environment variables
os.environ["AWS_ACCESS_KEY_ID"] = "o2QNQbVzFPi7UxuzPFLS"
os.environ["AWS_SECRET_ACCESS_KEY"] = "3jFr912WAWkaMnN5F6W3i16eKd7ikadYkwV1sMcf"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://10.1.1.118:9005"

# set the tracking uri
mlflow.set_tracking_uri('http://10.1.1.118:5005')

# set the experiment id
mlflow.set_experiment(experiment_id="0")

mlflow.autolog()
db = load_diabetes()

X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# Create and train models.
rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
rf.fit(X_train, y_train)

# Use the model to make predictions on the test dataset.
predictions = rf.predict(X_test)