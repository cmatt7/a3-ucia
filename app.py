import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

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