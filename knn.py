'''knn.py
K-Nearest Neighbors algorithm for classification
Hannah Soria
CS 251 Data Analysis Visualization, Spring 2021
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from palettable import cartocolors
import scipy


class KNN:
    '''K-Nearest Neighbors supervised learning algorithm'''
    def __init__(self, num_classes):
        '''KNN constructor

        TODO:
        - Add instance variable for `num_classes`
        '''
        # exemplars: ndarray. shape=(num_train_samps, num_features).
        #   Memorized training examples
        self.exemplars = None
        # classes: ndarray. shape=(num_train_samps,).
        #   Classes of memorized training examples
        self.classes = None

    def train(self, data, y):
        '''Train the KNN classifier on the data `data`, where training samples have corresponding
        class labels in `y`.

        Parameters:
        -----------
        data: ndarray. shape=(num_train_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_train_samps,). Corresponding class of each data sample.

        TODO:
        - Set the `exemplars` and `classes` instance variables such that the classifier memorizes
        the training data.
        '''
        self.exemplars = data
        self.classes = y

    def predict(self, data, k): # Grace Moberg helped me with this function
        '''Use the trained KNN classifier to predict the class label of each test sample in `data`.
        Determine class by voting: find the closest `k` training exemplars (training samples) and
        the class is the majority vote of the classes of these training exemplars.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
            Need not be the data used to train the network.
        k: int. Determines the neighborhood size of training points around each test sample used to
            make class predictions. In other words, how many training samples vote to determine the
            predicted class of a nearby test sample.

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_test_samps,). Predicted class of each test data
        sample.

        TODO:
        - Compute the distance from each test sample to all the training exemplars.
        - Among the closest `k` training exemplars to each test sample, count up how many belong
        to which class.
        - The predicted class of the test sample is the majority vote.
        '''
        total = np.zeros((data.shape[0], self.exemplars.shape[0])) # nd array of the num of samples, num of exemplar samples
        for i in range(data.shape[0]): # iterate through data
            dist = np.sqrt(np.sum((data[i,:] - self.exemplars)**2, axis=1)) # find the distance
            total[i] = dist # put distance into the total array

        predictions = np.zeros(data.shape[0]) #create an empty array for the classes predictions
        for i in range(data.shape[0]): 
            dist = total[i] # distance from the total array
            nn_index = dist.argsort()[:k] # nearest neighbor index sorted
            nn_label = self.classes[nn_index]  # nearest neighbor label from the classes
            vals, counts = np.unique(nn_label, return_counts = True) # sorted unique elements of np array

            index = np.argmax(counts) # get the index

            predictions[i] = vals[index] # get the predicted class put it i the array

        return predictions

    def accuracy(self, y, y_pred):
        '''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_sams,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_sams,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        float. Between 0 and 1. Proportion correct classification.

        NOTE: Can be done without any loops
        '''
        result = np.count_nonzero(y == y_pred)
        total = y.shape[0]
        
        return result/total

    def plot_predictions(self, k, n_sample_pts):
        '''Paints the data space in colors corresponding to which class the classifier would
         hypothetically assign to data samples appearing in each region.

        Parameters:
        -----------
        k: int. Determines the neighborhood size of training points around each test sample used to
            make class predictions. In other words, how many training samples vote to determine the
            predicted class of a nearby test sample.
        n_sample_pts: int.
            How many points to divide up the input data space into along the x and y axes to plug
            into KNN at which we are determining the predicted class. Think of this as regularly
            spaced 2D "fake data" that we generate and plug into KNN and get predictions at.

        TODO:
        - Pick a discrete/qualitative color scheme. We suggest, like in the clustering project, to
        use a ColorBrewer color palette. List of possible ones here:
        https://github.com/CartoDB/CartoColor/wiki/CARTOColor-Scheme-Names
            - An example: cartocolors.qualitative.Safe_4.mpl_colors
            - The 4 stands for the number of colors in the palette. For simplicity, you can assume
            that we're hard coding this at 4 for 4 classes.
        - Each ColorBrewer palette is a Python list. Wrap this in a `ListedColormap` object so that
        matplotlib can parse it (already imported above).
        - Make an ndarray of length `n_sample_pts` of regularly spaced points between -40 and +40.
        - Call `np.meshgrid` on your sampling vector to get the x and y coordinates of your 2D
        "fake data" sample points in the square region from [-40, 40] to [40, 40].
            - Example: x, y = np.meshgrid(samp_vec, samp_vec)
        - Combine your `x` and `y` sample coordinates into a single ndarray and reshape it so that
        you can plug it in as your `data` in self.predict.
            - Shape of `x` should be (n_sample_pts, n_sample_pts). You want to make your input to
            self.predict of shape=(n_sample_pts*n_sample_pts, 2).
        - Reshape the predicted classes (`y_pred`) in a square grid format for plotting in 2D.
        shape=(n_sample_pts, n_sample_pts).
        - Use the `plt.pcolormesh` function to create your plot. Use the `cmap` optional parameter
        to specify your discrete ColorBrewer color palette.
        - Add a colorbar to your plot
        '''
        cmap = ListedColormap(cartocolors.qualitative.Safe_4.mpl_colors) # cmap
        samp_vec = np.linspace(-40, 40, n_sample_pts) #sample vector
        x, y = np.meshgrid(samp_vec, samp_vec) # create np.meshgrid 
        xy = np.column_stack((x.flatten(), y.flatten())) # flatten data
        y_pred = self.predict(xy, k).reshape((n_sample_pts, n_sample_pts)) # reshape predicted classes
        fig, ax = plt.subplots(1, 1, figsize = (10, 5)) # create plot
        im = ax.pcolormesh(x, y, y_pred, cmap = cmap) # create plot
        fig.colorbar(im)
        plt.show





