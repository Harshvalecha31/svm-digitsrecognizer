from sklearn.datasets import load_digits
from sklearn import svm
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import time
#load the data
digits = load_digits()

# Here features are images and target variables are provided here..
#combined iterative form of image and labels..
image_labels = list(zip(digits.images,digits.target))

# to show the images of the dataset
"""for index,(image,label) in enumerate(image_labels[0:6]):
    plt.subplot(2,3,index+1)
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title('Target:%i'%label)
    
plt.show()
"""
n_features = len(digits.images)
#print(n_features) there are 1797 images!!
data = digits.images.reshape(n_features,-1)

# create the model !! ** when gamma was set to auto it provided only 50% accuracy but in 0.001 it is 95%
model = svm.SVC(gamma=0.001)

#new mthod of splitting the training and test data
traintestsplit = int(n_features*0.80)
# using 75% for training the data
# fitting the training and the test data to the model
model.fit(data[:traintestsplit],digits.target[:traintestsplit])
expected = digits.target[traintestsplit:]
prediction = model.predict(data[traintestsplit:])

im_viewer = n_features - traintestsplit
count =0
# predicting any particular image in the dataset
for i in range(im_viewer):
    plt.imshow(digits.images[i],cmap=plt.cm.gray_r,interpolation = 'nearest')
    print('Predicted image is:%i'%int(model.predict(digits.images[i].reshape(1,-1))))
    plt.show()
    count+=1
    time.sleep(0.1)

#apply accuracy_score and confusion_matrix to predict the results!!
print('No of images alssified:%i'%count)
print('confusion matrix:',confusion_matrix(expected,prediction))
print('accuracy score:',accuracy_score(expected,prediction))
