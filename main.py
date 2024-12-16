import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

### Loading Dataset
df = load_iris(as_frame=True)
df = df["frame"]
print(df)

### Splitting Dataset
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
print(y_train)

### Training the Model
model = SVC()
model.fit(X_train, y_train)

### Evaluating Model
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
f1 = accuracy_score(y_val, y_pred)
precision = accuracy_score(y_val, y_pred)
recall = accuracy_score(y_val, y_pred)
print(f"acc:{acc}\nf1:{f1}\nprecision:{precision}\nrecall:{recall}")

### Visualizing Confusion Matrix
cm = confusion_matrix(y_val, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()
