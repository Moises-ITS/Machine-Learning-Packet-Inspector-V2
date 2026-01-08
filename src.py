import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier

df_train = pd.read_csv('UNSW_NB15_training-set.csv')
df_test = pd.read_csv('UNSW_NB15_testing-set.csv')

df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)

df.drop(['id', 'attack_cat'], axis=1, errors='ignore', inplace=True)

X = df.drop('label', axis=1)
y = df['label']

#utilizing a different splitting method to capture those RARE cases and pay closer attention to those
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

num_col = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
enc = ['proto', 'state', 'service']
ratio = float(y_train.value_counts()[0]) / y_train.value_counts()[1] #similar to SMOTE but better

preprocessing = ColumnTransformer([('num', StandardScaler(), num_col), 
                 ('encode', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), enc)])
pipe = Pipeline(steps=[
    ('preprocessor', preprocessing),
    ('classifier', XGBClassifier(n_estimators=500, learning_rate=0.2, max_depth=6, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=ratio)) 
    #0.05 is actually smarter bc it made fewer total errors on normal traffic but 0.2 is prefered in cybersecurity because its safer and only has 1971 false negatives
])
model = pipe.fit(X_train, y_train)
prediction = pipe.predict(X_test)
#-----------------------------------------
#Classification & Accuracy Report
#----------------------------------------
print(f"Accuracy: {accuracy_score(y_test, prediction)}")
print("Classification Report:")
print(classification_report(y_test, prediction))

#-----------------------------------------
#Confusion Matrix
#-----------------------------------------
cm = confusion_matrix(y_test, prediction)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Attack'])
disp.plot(cmap="Blues", values_format='.2f')
plt.show()

#-----------------------------------------
#Paranoid Mode
#-----------------------------------------
# 1. Get the probabilities instead of the final 0/1 choice
# This gives two columns: [Prob of 0, Prob of 1]
probabilities = model.predict_proba(X_test)[:, 1]

# 2. Lower the threshold from 0.5 to 0.3
# "If you're 30% sure it's an attack, flag it."
paranoid_predictions = (probabilities > 0.3).astype(int)

# 3. Check the new Confusion Matrix
new_cm = confusion_matrix(y_test, paranoid_predictions)
print(f"Accuracy: {accuracy_score(y_test, paranoid_predictions)}")
print("New Confusion Matrix (Paranoid Mode):")
print(new_cm)
