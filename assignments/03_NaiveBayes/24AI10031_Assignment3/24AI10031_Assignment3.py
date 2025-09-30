# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sc
%matplotlib inline

# %% [markdown]
# # Loading Data

# %%
from sklearn.datasets import fetch_openml

data = fetch_openml(data_id=40499)
data

# %%
df = pd.DataFrame(data=data.data)
df['target'] = data.target

for col in df.columns[:-1]:
    df[col] = df[col].astype(np.float64)

df['target'] = df['target'].astype(np.int8)

df.head(5)

# %% [markdown]
# # Balanced Distribution of Data

# %%
bins = np.arange(0.5, 11.5 + 1)

plt.hist(df['target'], bins=bins, edgecolor='black')

plt.xticks(np.arange(1, 12))

plt.xlabel('Target')
plt.ylabel('Frequency')
plt.title('Distribution of target')
plt.show()

# %% [markdown]
# # Train Test Split

# %%
from sklearn.model_selection import train_test_split

X, y = df[df.columns[:-1]].to_numpy(), df[df.columns[-1]].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# %% [markdown]
# ### Distribution of data in training set and testing set

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

# Train
axes[0].hist(y_train, bins=bins, edgecolor='black')
axes[0].set_xticks(np.arange(1, 12))
axes[0].set_title("Train Target Distribution")
axes[0].set_xlabel("Target")
axes[0].set_ylabel("Frequency")

# Test
axes[1].hist(y_test, bins=bins, edgecolor='black', color='orange')
axes[1].set_xticks(np.arange(1, 12))
axes[1].set_title("Test Target Distribution")
axes[1].set_xlabel("Target")

plt.tight_layout()
plt.show()

# %% [markdown]
# # Feature Scaling: Not Required for Naive Bayes
# As We are calculating seperately the probability distribution for each feature
# So individual distribution doesnt impact each other

# %% [markdown]
# # Gaussian Naive Bayes Classifier Implementation

# %%
class GaussianNaiveBayesClassifier:
    def __init__(self):
        self.priors = {}
        self.likelihoods = {}

    def train(self, X, y):
        self.classes = np.unique(y)
        self.n_features = X.shape[1]
        self.priors = self._calculate_prior_proba(y)
        self.likelihoods = self._calculate_likelihoods(X, y)
    
    def _calculate_prior_proba(self, y):
        classes, cnt_per_class = np.unique(y, return_counts=True)
        total_cnt = len(y)
        return dict(zip(classes, cnt_per_class / total_cnt))
    
    def _calculate_likelihoods(self, X, y):
        likelihoods = {}
        for class_ in self.classes:
            X_class_ = X[y == class_]
            likelihoods[class_] = {
                "mean": np.mean(X_class_, axis=0),
                "var": np.var(X_class_, axis=0) + 1e-9
            }
        return likelihoods

    def _gaussian_pdf(self, x, mean, var):
        coeff = 1.0 / np.sqrt(2.0 * np.pi * var)
        exponent = np.exp(- (x - mean) ** 2 / (2 * var))
        return coeff * exponent

    def _predict_single(self, X_test_row):
        class_scores = {}
        for class_ in self.classes:
            prior_log = np.log(self.priors[class_])
            mean = self.likelihoods[class_]["mean"]
            var = self.likelihoods[class_]["var"]

            likelihood_log = np.sum(np.log(self._gaussian_pdf(X_test_row, mean, var)))
            class_scores[class_] = prior_log + likelihood_log

        return max(class_scores, key=class_scores.get)
    
    def predict(self, X_test):
        return np.array([self._predict_single(row) for row in X_test])

# %% [markdown]
# # Evaluation

# %%
impl_model = GaussianNaiveBayesClassifier()
impl_model.train(X_train, y_train)
impl_y_pred = impl_model.predict(X_test)

# %%
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

impl_accuracy = accuracy_score(y_true=y_test, y_pred=impl_y_pred)
print(f'model accuracy {impl_accuracy}')

# %%
impl_cnf_mat = confusion_matrix(y_true=y_test, y_pred=impl_y_pred)

impl_cnf_disp = ConfusionMatrixDisplay(confusion_matrix=impl_cnf_mat)
impl_cnf_disp.plot(cmap="Blues", values_format="d")
plt.title("Sklearn GaussianNB Confusion Matrix")
plt.show()

# %%
from sklearn.metrics import precision_score, recall_score

impl_precision = precision_score(y_true=y_test, y_pred=impl_y_pred, average="macro")
impl_recall    = recall_score(y_true=y_test, y_pred=impl_y_pred, average="macro")

print(f"Precision: {impl_precision}")
print(f"Recall: {impl_recall}")

# %%
impl_performance = {
    'accuracy': impl_accuracy,
    'confusion_matrix':impl_cnf_mat,
    'precision':impl_precision,
    'recall':impl_recall,
}

# %% [markdown]
# # Using Sklearn

# %%
from sklearn.naive_bayes import GaussianNB

sk_model = GaussianNB()
sk_model.fit(X_train, y_train)
sk_y_pred = sk_model.predict(X_test)

# %% [markdown]
# # Evaluation

# %%
sk_accuracy = accuracy_score(y_true=y_test, y_pred=sk_y_pred)
print(f'model accuracy {sk_accuracy}')

# %%
sk_cnf_mat = confusion_matrix(y_true=y_test, y_pred=sk_y_pred)

sk_cnf_disp = ConfusionMatrixDisplay(confusion_matrix=sk_cnf_mat)
sk_cnf_disp.plot(cmap="Blues", values_format="d")
plt.title("Sklearn GaussianNB Confusion Matrix")
plt.show()

# %%
sk_precision = precision_score(y_true=y_test, y_pred=sk_y_pred, average="macro")
sk_recall = recall_score(y_true=y_test, y_pred=sk_y_pred, average="macro")

print(f"Precision: {sk_precision}")
print(f"Recall: {sk_recall}")

# %%
sk_performance = {
    'accuracy': sk_accuracy,
    'confusion_matrix':sk_cnf_mat,
    'precision':sk_precision,
    'recall':sk_recall,
}

# %% [markdown]
# # Comparision

# %%
comp_df = pd.DataFrame(data={
    'my_implementation':impl_performance,
    'sklearn':sk_performance,
}).drop(index='confusion_matrix').round(4)
comp_df

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

ConfusionMatrixDisplay(sk_cnf_mat, display_labels=impl_model.classes).plot(
    ax=axes[0], cmap="Blues", values_format="d", colorbar=False
)
axes[0].set_title("Sklearn GaussianNB")

ConfusionMatrixDisplay(impl_cnf_mat, display_labels=impl_model.classes).plot(
    ax=axes[1], cmap="Blues", values_format="d", colorbar=False
)
axes[1].set_title("My Implementation")

plt.tight_layout()
plt.show()

# %%



