from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model(data):
    features = data[['SMA_50', 'SMA_200', 'RSI']]
    labels = data['Label']
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=False)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, predictions)}")
    
    return model
