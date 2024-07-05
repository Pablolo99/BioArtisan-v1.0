import pandas as pd
import matplotlib.pyplot as plt

# Load data from file
file_path = 'C:/Users/pablo/PycharmProjects/BioArtisan-v1.0/pred_models/PK/accuracy_results.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Separate data by model type
rf_data = df[df['Model'] == 'RandomForest']
svm_data = df[df['Model'] == 'SVM']
gbm_data = df[df['Model'] == 'GBM']
knn_data = df[df['Model'] == 'KNN']


# Accuracy plot
plt.figure(figsize=(20, 6))  # Larger figure size
plt.barh(rf_data['Hyperparameters'], rf_data['Average Accuracy'], height=0.6, color='darkblue', alpha=0.7, label='RandomForest')
plt.barh(svm_data['Hyperparameters'], svm_data['Average Accuracy'], height=0.6, color='lightblue', alpha=0.7, label='SVM')
plt.barh(gbm_data['Hyperparameters'], gbm_data['Average Accuracy'], height=0.6, color='orange', alpha=0.7, label='GBM')
plt.barh(knn_data['Hyperparameters'], knn_data['Average Accuracy'], height=0.6, color='lightcoral', alpha=0.7, label='KNN')
plt.xlabel('Average Accuracy')
plt.ylabel('Hyperparameters')
plt.title('Average Accuracy by Model')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()

# Plotting Average F1 Score
plt.figure(figsize=(20, 6))  # Larger figure size
plt.barh(rf_data['Hyperparameters'], rf_data['Average F1 Score'], height=0.6, color='darkblue', alpha=0.7, label='RandomForest')
plt.barh(svm_data['Hyperparameters'], svm_data['Average F1 Score'], height=0.6, color='lightblue', alpha=0.7, label='SVM')
plt.barh(gbm_data['Hyperparameters'], gbm_data['Average F1 Score'], height=0.6, color='orange', alpha=0.7, label='GBM')
plt.barh(knn_data['Hyperparameters'], knn_data['Average F1 Score'], height=0.6, color='lightcoral', alpha=0.7, label='KNN')
plt.xlabel('Average F1 Score')
plt.ylabel('Hyperparameters')
plt.title('Average F1 Score by Model')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()
