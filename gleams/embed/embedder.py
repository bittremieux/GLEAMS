import keras
from keras import backend as K
from keras import Input
from keras.layers import concatenate, Conv1D, Dense, Flatten, Lambda,\
    MaxPooling1D, Reshape
from keras.models import Model
from keras.optimizers import Adam

from gleams import config


def euclidean_distance(vects):
    """
    Euclidean distance between two vectors using Keras.

    Parameters
    ----------
    vects
        Two vectors between which to compute the Euclidean distance.

    Returns
    -------
    The Euclidean distance between the two given vectors.
    """
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    """
    Get the shape of the Euclidean distance output.

    Parameters
    ----------
    shapes
        Input shapes to the Euclidean distance calculation.

    Returns
    -------
    The shape of the Euclidean distance output.
    """
    shape1, shape2 = shapes
    return shape1[0], 1


def contrastive_loss(y_true, y_pred):
    """
    Contrastive loss function adapted from Hadsell et al. 2006.
    (http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)

    The contrastive loss is modified so that it takes a certainty that labels
    are correct into account. This helps the neural network to overcome
    incorrectly labeled instances.

    Parameters
    ----------
    y_true
        The true class labels.
    y_pred
        The predicted class labels.

    Returns
    -------
    The contrastive loss between the true and predicted class labels.
    """
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(config.margin - y_pred, 0))
    return K.mean(y_true * config.loss_label_certainty * square_pred +
                  (1 - y_true * config.loss_label_certainty) * margin_square)


class Embedder:
    """
    A spectrum embedder formed by a Siamese neural network.

    Inspired by the Keras MNIST Siamese network example:
    https://keras.io/examples/mnist_siamese/
    """

    def __init__(self, num_precursor_features: int, num_fragment_features: int,
                 num_ref_spectra_features: int, lr: float):
        """
        Instantiate the Embbeder based on the given number of input features.

        Parameters
        ----------
        num_precursor_features : int
            The number of input precursor features.
        num_fragment_features : int
            The number of input fragment features.
        num_ref_spectra_features : int
            The number of input reference spectra features.
        lr : float
            The learning rate for the Adam optimizer.
        """
        self.num_precursor_features = num_precursor_features
        self.num_fragment_features = num_fragment_features
        self.num_ref_spectra_features = num_ref_spectra_features
        self.lr = lr

        self.model = None

    def save(self, filename: str):
        """
        Save a model and its training status.

        Parameters
        ----------
        filename : str
            The file name to save the model.
        """
        if self.model is None:
            raise ValueError('The model hasn\'t been constructed yet')
        else:
            self.model.save(filename)

    def load(self, filename: str):
        """
        Load a saved model and its training status from the given file.

        Parameters
        ----------
        filename : str
            The file name of the saved model to be loaded.
        """
        self.model = keras.models.load_model(
            filename, custom_objects={'contrastive_loss': contrastive_loss})

    def _build_base_model(self) -> Model:
        """
        Construct the model architecture of a single arm of the Siamese
        network.

        The embedder network consists of the following elements:
        - Precursor features are processed using two fully-connected layers of
          dimensions 32 and 5. SELU activation is used.
        - The fragment features and reference spectra features are both
          processed through a single convolutional layer consisting of 30
          filters and kernel size 3, followed by a max pooling layer. SELU
          activation is used.
        - The output of all three elements is concatenated and processed using
          a single fully-connected layer of dimension 32.

        Returns
        -------
        Model
            The Keras neural network model for a single arm of the Siamese
            network that takes as input the features specified for this
            Embedder.
        """
        # Precursor features are processed through two dense layers.
        precursor_input = Input((self.num_precursor_features,))
        precursor_dense32 = (Dense(32, activation='selu',
                                   kernel_initializer='he_uniform')
                             (precursor_input))
        precursor_dense5 = (Dense(5, activation='selu',
                                  kernel_initializer='he_uniform')
                            (precursor_dense32))

        filters = 30
        kernel_size = 3
        strides = 1
        pool_size = 1
        pool_strides = 2
        # Fragment features are processed through a single convolutional and
        # max pooling layer.
        fragment_input = Input((self.num_fragment_features,))
        fragment_input_reshape =\
            Reshape((self.num_fragment_features, 1))(fragment_input)
        fragment_conv = Conv1D(filters, kernel_size, strides=strides,
                               activation='selu')(fragment_input_reshape)
        fragment_maxpool = MaxPooling1D(pool_size, pool_strides)(fragment_conv)
        fragment_output = Flatten()(fragment_maxpool)

        # Reference spectra features are processed through a single
        # convolutional and max pooling layer.
        ref_spectra_input = Input((self.num_ref_spectra_features,))
        ref_spectra_input_reshape =\
            Reshape((self.num_ref_spectra_features, 1))(ref_spectra_input)
        ref_spectra_conv = Conv1D(filters, kernel_size, strides=strides,
                                  activation='selu')(ref_spectra_input_reshape)
        ref_spectra_maxpool = (MaxPooling1D(pool_size, pool_strides)
                               (ref_spectra_conv))
        ref_spectra_output = Flatten()(ref_spectra_maxpool)

        # Combine all outputs and add a final dense layer.
        output_layer = (Dense(32, activation='selu',
                              kernel_initializer='he_uniform')
                        (concatenate([precursor_dense5, fragment_output,
                                      ref_spectra_output])))

        return Model(inputs=[precursor_input, fragment_input,
                             ref_spectra_input],
                     outputs=[output_layer])

    def build_siamese_model(self):
        """
        Build the Embedder's Siamese network and compile it to optimize the
        contrastive loss using Adam.

        Both arms of the Siamese network will use the same base network, i.e.
        the weights are tied between both arms.
        """
        # Both arms of the Siamese network use the same model, i.e. the weights
        # are tied.
        base_model = self._build_base_model()
        input_left = [Input((self.num_precursor_features,)),
                      Input((self.num_fragment_features,)),
                      Input((self.num_ref_spectra_features,))]
        input_right = [Input((self.num_precursor_features,)),
                       Input((self.num_fragment_features,)),
                       Input((self.num_ref_spectra_features,))]
        output_left = base_model(input_left)
        output_right = base_model(input_right)

        # Euclidean distance between two embeddings.
        distance = (Lambda(euclidean_distance, eucl_dist_output_shape)
                    ([output_left, output_right]))

        # Train using Adam to optimize the contrastive loss.
        self.model = Model(inputs=[*input_left, *input_right],
                           outputs=distance)
        self.model.compile(Adam(self.lr), contrastive_loss)

    def train(self, x_train, y_train, batch_size: int, num_epochs: int,
              x_val=None, y_val=None):
        if self.model is None:
            raise ValueError("The model hasn't been constructed yet")

        # TODO: Callbacks.
        self.model.fit(
            x_train, y_train, batch_size, num_epochs,
            validation_data=((x_val, y_val) if x_val is not None and
                                               y_val is not None else None))
