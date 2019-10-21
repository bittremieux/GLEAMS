import logging
import os
import warnings
from typing import List

import keras
import numpy as np
from keras import backend as K
from keras import Input
from keras.callbacks import CSVLogger, TensorBoard
from keras.layers import concatenate, Conv1D, Dense, Flatten, Lambda,\
    MaxPooling1D, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import multi_gpu_utils
from sklearn.metrics import auc, roc_curve

from gleams import config
from gleams.nn import data_generator


logger = logging.getLogger('gleams')


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
                 num_ref_spectra_features: int, lr: float,
                 filename: str = 'gleams.hdf5'):
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
        filename : str
            Filename to save the trained Keras model.
        """
        self.num_precursor_features = num_precursor_features
        self.num_fragment_features = num_fragment_features
        self.num_ref_spectra_features = num_ref_spectra_features
        self.lr = lr
        self.filename = filename

        self.siamese_model = self.siamese_model_parallel = None

    def _get_base_model(self) -> Model:
        """
        Get the base embedder model (i.e. a single arm of the Siamese model).

        Returns
        -------
        Model
            The embedder model.
        """
        if self.siamese_model_parallel is None:
            raise ValueError("The Siamese model hasn't been compiled yet")
        else:
            return self.siamese_model_parallel.get_layer('base_model')

    def _init_multi_gpu(self) -> Model:
        """
        Replicate the Siamese model over multiple GPUs if possible.

        If multiple GPUs are used the learning rate is multiplied accordingly.

        Returns
        -------
        Model
            The multi-GPU version of the Siamese model.
        """
        if self.siamese_model is None:
            raise ValueError("The Siamese model hasn't been compiled yet")
        else:
            # Use multiple GPUs if available.
            try:
                available_devices = [
                    multi_gpu_utils._normalize_device_name(name)
                    for name in multi_gpu_utils._get_available_devices()]
                num_gpus = len([x for x in available_devices if '/gpu' in x])
                model = multi_gpu_utils.multi_gpu_model(self.siamese_model,
                                                        gpus=num_gpus,
                                                        cpu_relocation=True)
                logger.info('Parallelizing the Siamese model over %d GPUs',
                            num_gpus)
            except ValueError:
                model = self.siamese_model
                logger.info('Running the Siamese model on a single GPU')
            return model

    def save(self) -> None:
        """
        Save the Siamese model and its training status.
        """
        if self.siamese_model is None:
            raise ValueError("The Siamese model hasn't been compiled yet")
        else:
            self.siamese_model.save(self.filename)

    def load(self) -> None:
        """
        Load the saved Siamese model and its training status from the given
        file.
        """
        self.siamese_model = keras.models.load_model(
            self.filename,
            custom_objects={'contrastive_loss': contrastive_loss})
        self.siamese_model_parallel = self._init_multi_gpu()

    def _build_base_model(self) -> Model:
        """
        Construct the embedder model (i.e. a single arm of the Siamese model).

        The embedder model consists of the following elements:
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
            The embedder model that takes as input the features specified.
        """
        # Precursor features are processed through two dense layers.
        precursor_input = Input((self.num_precursor_features,),
                                name='input_precursor')
        precursor_dense1 = (Dense(32, activation='selu',
                                  kernel_initializer='he_uniform',
                                  name='precursor_dense_1')
                            (precursor_input))
        precursor_dense2 = (Dense(5, activation='selu',
                                  kernel_initializer='he_uniform',
                                  name='precursor_dense_2')
                            (precursor_dense1))

        filters = 30
        kernel_size = 3
        strides = 1
        pool_size = 1
        pool_strides = 2
        # Fragment features are processed through a single convolutional and
        # max pooling layer.
        fragment_input = Input((self.num_fragment_features,),
                               name='input_fragment')
        fragment_input_reshape = (Reshape((self.num_fragment_features, 1),
                                          name='fragment_input_reshape')
                                  (fragment_input))
        fragment_conv_1 = (Conv1D(filters, kernel_size, strides=strides,
                                  activation='selu', name='fragment_conv_1')
                           (fragment_input_reshape))
        fragment_pool_1 = (MaxPooling1D(pool_size, pool_strides,
                                        name='fragment_pool_1')
                           (fragment_conv_1))
        fragment_output = Flatten(name='fragment_flatten')(fragment_pool_1)

        # Reference spectra features are processed through a single
        # convolutional and max pooling layer.
        ref_spectra_input = Input((self.num_ref_spectra_features,),
                                  name='input_ref_spectra')
        ref_spectra_input_reshape = (Reshape((
            self.num_ref_spectra_features, 1),
            name='ref_spectra_input_reshape')(ref_spectra_input))
        ref_spectra_conv_1 = (Conv1D(filters, kernel_size, strides=strides,
                                     activation='selu',
                                     name='ref_spectra_conv_1')
                              (ref_spectra_input_reshape))
        ref_spectra_pool_1 = (MaxPooling1D(pool_size, pool_strides,
                                           name='ref_spectra_pool_1')
                              (ref_spectra_conv_1))
        ref_spectra_output = (Flatten(name='ref_spectra_flatten')
                              (ref_spectra_pool_1))

        # Combine all outputs and add a final dense layer.
        output_layer = (Dense(32, activation='selu',
                              kernel_initializer='he_uniform', name='output')
                        (concatenate([precursor_dense2, fragment_output,
                                      ref_spectra_output])))

        return Model(inputs=[precursor_input, fragment_input,
                             ref_spectra_input],
                     outputs=[output_layer], name='base_model')

    def build_siamese_model(self) -> None:
        """
        Build the Siamese model and compile it to optimize the contrastive loss
        using Adam.

        Both arms of the Siamese network will use the same base model, i.e.
        the weights are tied between both arms.
        """
        # Both arms of the Siamese network use the same model,
        # i.e. the weights are tied.
        base_model = self._build_base_model()
        input_left = [Input((self.num_precursor_features,),
                            name='input_precursor_left'),
                      Input((self.num_fragment_features,),
                            name='input_fragment_left'),
                      Input((self.num_ref_spectra_features,),
                            name='input_ref_spectra_left')]
        input_right = [Input((self.num_precursor_features,),
                             name='input_precursor_right'),
                       Input((self.num_fragment_features,),
                             name='input_fragment_right'),
                       Input((self.num_ref_spectra_features,),
                             name='input_ref_spectra_right')]
        output_left = base_model(input_left)
        output_right = base_model(input_right)

        # Euclidean distance between two embeddings.
        distance = (Lambda(euclidean_distance, eucl_dist_output_shape,
                           name='embedding_euclidean_distance')
                    ([output_left, output_right]))

        # Train using Adam to optimize the contrastive loss.
        self.siamese_model = Model(inputs=[*input_left, *input_right],
                                   outputs=distance, name='siamese_model')
        self.siamese_model_parallel = self._init_multi_gpu()
        self.siamese_model_parallel.compile(Adam(self.lr), contrastive_loss)

    def train(self, train_generator: data_generator.PairSequence,
              steps_per_epoch: int = None, num_epochs: int = 1,
              val_generator: data_generator.PairSequence = None) -> None:
        """
        Train the Siamese model.

        Parameters
        ----------
        train_generator : data_generator.PairSequence
            The training data generator.
        steps_per_epoch : int
             Total number of in each epoch. Useful to record the validation
             loss at specific intervals.
        num_epochs : int
            The number of epochs for which training occurs.
        val_generator : data_generator.PairSequence
            The validation data generator.
        """
        if self.siamese_model_parallel is None:
            raise ValueError("The Siamese model hasn't been compiled yet")

        filename, ext = os.path.splitext(self.filename)
        filename_log = f'{filename}.log'
        # CrocHistory has to be added after CSVLogger because it uses the same
        # log file.
        callbacks = [ModelCheckpointMultiGpuCompatible(
            self.siamese_model, filename + '.epoch{epoch:03d}' + ext),
                     CrocHistory(val_generator, filename_log),
                     CSVLogger(filename_log),
                     TensorBoard('/tmp/gleams', update_freq='batch')]
        self.siamese_model_parallel.fit_generator(
            train_generator, steps_per_epoch=steps_per_epoch,
            epochs=num_epochs, callbacks=callbacks,
            validation_data=val_generator)

    def embed(self, x: List[np.ndarray]) -> np.ndarray:
        """
        Transform samples using the embedder model.

        Parameters
        ----------
        x : List[np.ndarray]
            The input samples as a list of length three representing the
            precursor features, fragment features, and reference spectra
            features.

        Returns
        -------
        np.ndarray
            The embeddings of the given samples.
        """
        return self._get_base_model().predict(x)


class ModelCheckpointMultiGpuCompatible(keras.callbacks.Callback):

    def __init__(self, model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super().__init__()
        self.model_to_save = model
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, fallback to '
                          'auto mode.' % mode, RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available,'
                                  ' skipping.' % self.monitor, RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to '
                                  '%0.5f, saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model_to_save.save_weights(filepath,
                                                            overwrite=True)
                        else:
                            self.model_to_save.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from '
                                  '%0.5f' % (epoch + 1, self.monitor,
                                             self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s'
                          % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model_to_save.save_weights(filepath, overwrite=True)
                else:
                    self.model_to_save.save(filepath, overwrite=True)


class CrocHistory(keras.callbacks.Callback):
    """
    Track the AUC CROC on the validation data after each epoch ends.
    """

    # Alpha = 14 maps x = 0.05 to 0.5.
    alpha = 14

    def __init__(self, pair_generator, log_filename=None):
        super().__init__()

        self.pair_generator = pair_generator
        self.log_filename = log_filename

    def on_epoch_end(self, epoch, logs=None):
        if self.pair_generator is not None:
            epoch_croc_aucs = []
            for batch_i in range(len(self.pair_generator)):
                batch_x, batch_y = self.pair_generator[batch_i]
                y_pred = self.model.predict(batch_x)
                fpr, tpr, _ = roc_curve(batch_y, 1 - y_pred / y_pred.max())
                # Exponential CROC transformation from Swamidass et al. 2010.
                croc_fpr = ((1 - np.exp(-self.alpha * fpr)) /
                            (1 - np.exp(-self.alpha)))
                epoch_croc_aucs.append(auc(croc_fpr, tpr))
            logs['croc_auc'] = np.mean(epoch_croc_aucs)
