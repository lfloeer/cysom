#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

import numpy as np

cimport numpy as np
cimport cython

cdef extern from 'math.h':
    float sqrt(float a) nogil
    float exp(float a) nogil

cdef class SOMBase:
    """
    Base class for self-organizing maps that implements both batch and on-line
    training.
    """

    cdef:
        # arrays
        np.ndarray _som_weights
        np.ndarray _som_weight_diff
        np.ndarray _map_shape
        np.ndarray _map_coordinates
        
        # memoryviews
        float[:,:] _map_coordinates_v
        float[:,:] _som_weights_v
        float[:,:] _som_weights_diff_v
        int[:] _map_shape_v
        
        # size descriptors
        int _n_dim
        int _som_size

    def __init__(self, map_shape, n_dim):
        """
        Parameters
        ----------
        map_shape : tuple of ints
            The shape of the self-organizing map.
        n_dim : integer
            The dimensionality of each vector in the map.
        """

        self._map_shape = np.array(map_shape, dtype=np.int32)
        self._n_dim = n_dim
        self._som_size = np.prod(self._map_shape)
        
        self._som_weights = np.random.random((self._som_size, self._n_dim)).astype(np.float32)
        self._som_weight_diff = self._som_weights.copy()

        # self._map_coordinates[:,i] has the coordinates for node i
        self._map_coordinates = np.indices(
            self._map_shape, dtype=np.float32).reshape((self._map_shape.shape[0], self._som_size))

        self._som_weights_v = self._som_weights
        self._map_shape_v = self._map_shape
        self._som_weights_diff_v = self._som_weight_diff
        self._map_coordinates_v = self._map_coordinates

    
    property som_weights:
        """
        The current weights of the SOM.
        """

        def __get__(self):
            return self._som_weights.reshape(tuple(self._map_shape) + (-1,))


    property map_shape:
        """
        The shape of the SOM.
        """

        def __get__(self):
            return self._map_shape


    property n_dim:
        """
        The dimensionality of the input data-space.
        """

        def __get__(self):
            return self._n_dim
    

    def train_sample(self, float[:] sample, float learning_rate, float neighborhood_size):
        """
        Train a single data sample into the self-organizing map with
        the given learning_rate and neighborhood_size.

        Parameters
        ----------
        sample : (n_dim,) array_like
            1D array describing a single training example. Has to be of n_dim
            length.
        learning_rate : float
            The learning rate with which the sample is trained.
        neighborhood_size : float
            The size of the neighborhood.
        """

        cdef:
            float squared_neighborhood_size
            float squared_neuron_distance, neighborhood
            int winning_neuron
            int i,j

        squared_neighborhood_size = neighborhood_size * neighborhood_size

        winning_neuron = self.get_winning_neuron(sample)

        # Update weights
        for i in range(self._som_size):
            squared_neuron_distance = self.get_neuron_distance(winning_neuron, i)
            neighborhood = self.get_neighborhood(squared_neuron_distance, squared_neighborhood_size)
            for j in range(self._n_dim):
                self._som_weights_v[i,j] += learning_rate * neighborhood * self._som_weights_diff_v[i,j]


    def train_batch(self, float[:,:] samples, float neighborhood_size):
        """
        Train a batch of samples into the self-organizing map with
        the given neighborhood size.

        Parameters
        ----------
        sample : (n_samples, n_dim) array_like
            2D array describing multiple training examples.
        neighborhood_size : float
            The size of the neighborhood.
        """

        cdef:
            float squared_neighborhood_size = neighborhood_size ** 2
            float neighborhood
            np.ndarray normalizations, averages
            int winning_neuron
            int i,j,k

        averages = np.zeros((self._som_size, self._n_dim), dtype=np.float32)
        normalizations = np.zeros(self._som_size, dtype=np.float32)

        cdef float[:,:] averages_v = averages
        cdef float[:] normalizations_v = normalizations

        for i in range(samples.shape[0]):
            winning_neuron = self.get_winning_neuron(samples[i])

            for j in range(self._som_size):
                squared_neuron_distance = self.get_neuron_distance(winning_neuron, j)
                neighborhood = self.get_neighborhood(squared_neuron_distance, squared_neighborhood_size)
                
                for k in range(self._n_dim):
                    averages_v[j,k] += neighborhood * self._som_weights_diff_v[j,k]
                
                normalizations_v[j] += neighborhood

        for j in range(self._som_size):
            for k in range(self._n_dim):
                self._som_weights_v[j,k] = averages_v[j,k] / normalizations_v[j]


    def classify_sample(self, float[:] sample):
        """
        Return the coordinates of the winning for
        the given sample.

        Parameters
        ----------
        sample : (n_dim,) array_like
            1D array describing a single training example. Has to be of n_dim
            length.
        

        Returns
        -------
        coordinates : array
            The coordinates of the winning neuron.
        """

        cdef:
            int winning_neuron

        winning_neuron = self.get_winning_neuron(sample)

        return self._map_coordinates[:,winning_neuron]


    cdef int get_winning_neuron(self, float[:] sample):
        """
        Determine the winning neuron for the given data sample
        and return the flat index.
        """

        cdef:
            float min_weight_distance = 1e30
            float weight_distance
            int winning_neuron = -1
            int i,j

        for i in range(self._som_size):
            weight_distance = 0
            for j in range(self._n_dim):
                self._som_weights_diff_v[i,j] = sample[j] - self._som_weights_v[i,j]
                weight_distance += self._som_weights_diff_v[i,j] * self._som_weights_diff_v[i,j]

            if weight_distance < min_weight_distance:
                min_weight_distance = weight_distance
                winning_neuron = i

        return winning_neuron


    cdef float get_neuron_distance(self, int first, int second):
        """
        Compute the **squared** distance between neuron _first_ and neuron _second_.
        Override this function to change the map topology.
        """

        cdef:
            float distance, difference
            int i

        distance = 0

        for i in range(self._map_coordinates_v.shape[0]):
            difference = self._map_coordinates_v[i, first] - self._map_coordinates_v[i, second]
            distance += difference * difference

        return distance


    cdef float get_neighborhood(self, float squared_distance, float squared_neighborhood_size):
        """
        Return the neighborhood measure for a given squared distance
        between two neurons.
        """
        return exp(-0.5 * squared_distance / squared_neighborhood_size)
