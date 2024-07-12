import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import dump

data = pd.read_csv('Data_image_features.csv')

X = data.drop('label', axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



model = MLPClassifier(
    hidden_layer_sizes=(128,64),  
    activation='relu',                
    solver='adam',                    
    alpha=0.0001,                      
    learning_rate='adaptive',         
    max_iter=10000,                    
    random_state=55
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)

dump(model, 'trained_model.joblib')
