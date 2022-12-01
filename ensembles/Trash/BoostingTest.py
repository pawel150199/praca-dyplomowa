import os
from sklearn.naive_bayes import GaussianNB
from Boosting import Boosting
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import  matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Data
X, y = make_classification(
    n_samples=10000,
    n_classes=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    flip_y=0.1,
    weights=[0.9, 0.1],
    random_state=1410
)

# Dataset divisiom
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Boosting
Boosting = Boosting(base_estimator=GaussianNB(), M=300)
Boosting.fit(X_train, y_train)
y_pred_boosting = Boosting.predict(X_test)
print(y_pred_boosting)
acc_boosting = accuracy_score(y_test, y_pred_boosting)
print("Accuray score for Boosting %.2f" % (acc_boosting))

# Undersampled Boosting
UndersampledBoosting = UBO(base_estimator=GaussianNB(), M=300)
UndersampledBoosting.fit(X_train, y_train)
y_pred_undersampled = UndersampledBoosting.predict(X_test)
acc_undersampled_boosting = accuracy_score(y_test, y_pred_undersampled)
print("Accuray score for Undersampled Boosting %.2f" % (acc_undersampled_boosting))

# Oversampled Boosting
OversampledBoosting = UBO(base_estimator=GaussianNB(), M=300)
OversampledBoosting.fit(X_train, y_train)
y_pred_oversampled = OversampledBoosting.predict(X_test)
acc_oversampled_boosting = accuracy_score(y_test, y_pred_oversampled)
print("Accuray score for Oversampled Boosting %.2f" % (acc_oversampled_boosting))

# Plot
fig, ax = plt.subplots(2,2, figsize=(10,5))

# Boosting
ax[0,0].plot(Boosting.training_errors)
#ax[0.0].axhline(y=0.5, xmin=0, xmax=400, colors = "red", linewidth=0.5)
ax[0,0].set_title('Training error rates by stump Boosting')
ax[0,0].set_xlabel('Stump')
ax[0,0].set_ylabel('Errors')

# Undersampled Boosting
ax[0,1].plot(UndersampledBoosting.training_errors)
#ax[0.1].axhline(y=0.5, xmin=0, xmax=400, colors = 'red', linestyles='dashed', linewidth=0.5)
ax[0,1].set_title('Training error rates by stump UndersampledBoosting')
ax[0,1].set_xlabel('Stump')
ax[0,1].set_ylabel('Errors')

# Oversampled Boosting
ax[1,0].plot(OversampledBoosting.training_errors)
#ax[1.0].axhline(y=0.5, xmin=0, xmax=400, colors = 'red', linestyles='dashed', linewidth=0.5)
ax[1,0].set_title('Training error rates by stump OversamplesBoosting')
ax[1,0].set_xlabel('Stump')
ax[1,0].set_ylabel('Errors')


os.chdir("../images")
plt.tight_layout()
plt.savefig("BoostingErrorRate.png")
