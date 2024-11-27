import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import numpy as np
# NB
from sklearn.naive_bayes import GaussianNB
# Rede Neural
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

data = pd.read_csv('C:/Users/mathe/.cache/kagglehub/datasets/bhadramohit/social-media-usage-datasetapplications/versions/1/social_media_usage.csv')

result = data.head()

print('Tratativa de dados\n\n' + str(result) + '\n\n')

missing_data = data.isnull().sum()

statistical_summary = data.describe()

print('Valores ausentes\n\n' + str(missing_data) + '\n\n') 
print('Resumo estatístico\n\n' + str(statistical_summary) + '\n\n')

data['Engagement_Level'] = pd.cut(data['Daily_Minutes_Spent'],
                                  bins=[0, 100, 300, 500],
                                  labels=['Low', 'Medium', 'High'])

engagement_distribution = data['Engagement_Level'].value_counts()

features = data[['Posts_Per_Day', 'Likes_Per_Day', 'Follows_Per_Day']]

target = data['Engagement_Level']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

print('Distribuição de engagamento\n\n' + str(engagement_distribution) + '\n\n' + str(X_train.shape), str(X_test.shape) + '\n\n')

dt_classifier = DecisionTreeClassifier(random_state=42)

dt_classifier.fit(X_train, y_train)

y_pred_dt = dt_classifier.predict(X_test)

dt_performance = classification_report(y_test, y_pred_dt)

print('Decision Tree Performance\n', dt_performance)

class_prior_high = len(data[data['Engagement_Level'] == 'High']) / len(data)
class_prior_medium = len(data[data['Engagement_Level'] == 'Medium']) / len(data)
class_prior_low = len(data[data['Engagement_Level'] == 'Low']) / len(data)

nb_classifier = GaussianNB(priors=[class_prior_low, class_prior_medium, class_prior_high])

nb_classifier.fit(X_train, y_train)

y_pred_nb = nb_classifier.predict(X_test)

nb_performance = classification_report(y_test, y_pred_nb, zero_division=0)
print('Naive Baynes Performance\n', nb_performance)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

y_train_categorical = to_categorical(y_train_encoded)
y_test_categorical = to_categorical(y_test_encoded)

model = Sequential([
    Dense(64, input_shape=(X_train_scaled.shape[1],), activation='relu'),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_scaled, y_train_categorical, epochs=50, batch_size=10, validation_data=(X_test_scaled, y_test_categorical))

loss, accuracy = model.evaluate(X_test_scaled, y_test_categorical)
print(f'Accuracy: {accuracy*100:.2f}%')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

predictions = model.predict(X_test_scaled)
predicted_classes = np.argmax(predictions, axis=1)

print(classification_report(y_test_encoded, predicted_classes, target_names=label_encoder.classes_))