from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = datasets.load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Train the model with K=3
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(x_train, y_train)

# Make predictions on the test data
y_pred = classifier.predict(x_test)

# Print confusion matrix and accuracy metrics
print('Confusion Matrix')
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
print('Accuracy Metrics')
print(classification_report(y_test, y_pred))

# Set up the matplotlib figure with subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 7))

# Plot the initial clusters (True labels)
axs[0].scatter(x[:, 0], x[:, 1], c=y, cmap='viridis', marker='o', edgecolor='k')
axs[0].set_xlabel(iris.feature_names[0])
axs[0].set_ylabel(iris.feature_names[1])
axs[0].set_title('Initial Clusters with True Labels')

# Plot the heatmap of the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axs[1], cbar=False)
axs[1].set_xlabel('Predicted Labels')
axs[1].set_ylabel('True Labels')
axs[1].set_title('Confusion Matrix Heatmap')

# Show the plots
plt.tight_layout()
plt.show()