# Carnegie Classification
## About this project
Carnegie Classifications provide a way to make meaningful comparisons among similar institutions according to objective criterias.
Those universities are further classified by their level of research activity measured through several factors, such as research expenditures, number doctorate degrees awarded, number of research-focused faculty. This classification will group similar universities together based on their values with respect to the designated variables. Ideally, universities in a group should be very similar and groups should be dissimilar as possible from one another. The classification system is based on two separate indices calculated using Principal Component Analysis (PCA). The first index is based on a set of aggregate variables and the other on a set of per-capita
metrics.
This analysis tends to re-creates the calculation of the classifications using scikit-learn.

### Data:
The data used here were obtained from the Carnegie Classifications website at http://carnegieclassifications.iu.edu/
### Method:
The Carnegie Classifications are built using Principal Components Analysis (PCA). So for this analysis I gonna use PCA to recreate the Carnegie Classifications
In Python, calculation of the PCA can be done using scikit-learn.
