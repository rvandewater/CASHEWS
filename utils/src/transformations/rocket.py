from aeon.transformations.collection.convolution_based import MiniRocketMultivariate


class WearableDataTransformer:
    def __init__(self, num_kernels, verbose):
        self.model = MiniRocketMultivariate(num_kernels=num_kernels, verbose=verbose)

    def fit(self, data):
        self.model.fit(data)

    def transform(self, data):
        return self.model.transform(data)
