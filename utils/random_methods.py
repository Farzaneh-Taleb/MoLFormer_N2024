import numpy as np
from sklearn.linear_model import Ridge

#find element-wise product of the rows of the two matrices.
#but these odors differ at a fine-grained level because peppermint has an additional (but nondominant) ‘spicy’ note
#we unwrapped Pearson’s correlation of odor ratings in the perceptual space in the form of an element-wise product of ratings
def find_finegrained_similarity(arr1, arr2):
    return np.multiply(arr1, arr2)


#find element-wise product of two matrices and return the maximul value of each row and the index to the maximum value.
#Coarse similarity between two odors was defined as the element-wise product of their most dominant perceptual descriptor rating
#capturing only the extent to which two odors were rated highly on one common dominant perceptual descriptor
#For example, peppermint and spearmint are similar at a coarse level because they are both rated highly on the ‘cool’ dimension,
#We then identified the perceptual feature for which the product of perceptual ratings for the two odors was the highest.
# We defined pc as the product of ratings for the identified feature
#we assumed that, if two odors belong to the same coarse dimension, then the odor ratings on the feature most related to their mutual dimension would have a high product.

def find_coarse_similarity(arr1, arr2):
    product = find_finegrained_similarity(arr1, arr2)
    max_product = np.max(product, axis=1)
    max_index = np.argmax(product, axis=1)
    return max_product, max_index



#write a test function to test the functions
def test_find_coarse_similarity():
    arr1 = np.array([[1, 2, 3], [4, 5, 6]])
    arr2 = np.array([[1, 2, 3], [4, 5, 6]])
    max_product, max_index = find_coarse_similarity(arr1, arr2)
    assert np.all(max_product == [3, 6])
    assert np.all(max_index == [2, 2])
    print("All tests passed.")



# We train a linear ridge regression model to predict the representation of the molecules from the chemistry property. We then remove the chemistry property from the representation of the molecules by subtracting the predicted chemistry represenation from the real representation of the molecules.
def remove_chemistry_representation(chemistry_property_X, representations_y):
    #todo Train-test split how to do it?
    #todo cross-validation to find the best alpha
    reg = Ridge(alpha=1.0)
    reg.fit(chemistry_property_X, representations_y)
    predicted_representation_y = reg.predict(chemistry_property_X)
    return representations_y - predicted_representation_y


#write a test function to test the functions
def test_remove_chemistry_representation():
    chemistry_property_X = np.array([[1, 2, 3], [4, 5, 6]])
    representations_y = np.array([[1, 2, 3], [4, 5, 6]])
    result = remove_chemistry_representation(chemistry_property_X, representations_y)
    assert np.all(result == np.array([[0, 0, 0], [0, 0, 0]]))
    print("All tests passed.")



# test_find_coarse_similarity()
# test_remove_chemistry_representation()