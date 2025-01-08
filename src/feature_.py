from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import pandas as pd

# Load test data and predictions
engineered_data = pd.read_csv("data/final_engineered_features.csv")
X_test = engineered_data.drop(['anomalous'], axis=1).iloc[-1400:]  # Adjust indices as per your split
y_test = engineered_data['anomalous'].iloc[-1400:]

# Load trained model
rf = joblib.load('models/random_forest_classifier.joblib')
y_pred = rf.predict(X_test)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Random Forest')
plt.show()
