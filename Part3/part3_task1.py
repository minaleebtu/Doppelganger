import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the new sorted dataset
# new = pd.read_csv('Joined_1.csv')
new = pd.read_csv('Joinedv1.csv')


# drop the unnecessary columns
# datdrop = new.drop(['Unnamed: 0','Unnamed: 0.1' ,'Unnamed: 0.1.1','username', 'content', 'Label'], axis = 1)
datdrop = new.drop(['Unnamed: 0', 'username', 'comment', 'Label'], axis = 1)
# print("\n",datdrop)

# Just wanna see the value of each authors
new_value = new['username'].value_counts()
# print("\n",new_value)

# Convert the dataset into an array
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
dat_array = datdrop.to_numpy()
print("Features Vector\n", dat_array)

# Standardizing the data
scaler = StandardScaler()
dat_standardized = scaler.fit_transform(dat_array)
# print(dat_standardized)

# Calculate the Covariance Matrix
COVMAT = np.cov(dat_standardized.T)
print("Covariance Matrix:\n", COVMAT)

# Calculate the Eigenvalues and Eigenvectors
eigvals, eigvecs = np.linalg.eig(COVMAT)

print("Eigenvalues : \n", eigvals)
print("Eigenvectors : \n", eigvecs)


# Calculate the PCA and extract the features
pd.options.display.width = 0
pca = PCA(n_components = 0.999)
DATA_PCA = pca.fit_transform(dat_array)
print("PCA Data : \n", DATA_PCA)

# See the difference shape/dimension before and after extraction
print("Shape Transformation : \n", "\nBefore Extraction :", dat_array.shape[1], "\nAfter Extraction Using PCA : ", DATA_PCA.shape[1])

datfr = pd.DataFrame(DATA_PCA)

print(datfr)

