.. _optimizer_settings:
.. _learning_rate_scheduling_settings:

==================
Optimizer Settings
==================

.. note::
    To define the update algorithm, set the parameter ``optimizer`` to a dictionary
    and define the type by setting ``class``.
    All available optimizers and their parameters can be found :ref:`here <optimizer>`.
    Setting the learning rate should not set in the dict, but rather separately.
    If no updater is specified, plain SGD is used.

    The learning rate control scheme is set with ``learning_rate_control``,
    and many possible settings are available for the different control schemes.
    For the default values have a look at :mod:`returnn.learning_rate_control`.

.. warning::
    RETURNN will override the optimizer epsilon with 1e-16 if not specified otherwise, this can lead to unwanted
    behaviour when assuming a default epsilon of e.g. 1e-8 for Adam.


accum_grad_multiple_step
    An integer specifying the number of updates to stack the gradient, called "gradient accumulation".

dynamic_learning_rate
    This can be set as either a dictionary or a function.

    When setting a dictionary, a cyclic learning rate can be implemented by setting the parameters
    ``interval`` and ``decay``.
    The global learning rate is then multiplied by ``decay ** (global_step % interval)``

    When using a custom function, the passed parameters are ``network``, `global_train_step`` and ``learning_rate``.
    Do not forget to mark the parameters as variable args and add ``**kwargs`` to keep the config
    compatible to future changes.

    An example for Noam-style learning rate scheduling would be:

    .. code-block:: python

        learning_rate = 1  # can be higher, reasonable values may be up to 10 or even more
        learning_rate_control = "constant"

        def noam(n, warmup_n, model_d):
          """
          Noam style learning rate scheduling

          (k is identical to the global learning rate)

          :param int|float|tf.Tensor n:
          :param int|float|tf.Tensor warmup_n:
          :param int|float|tf.Tensor model_d:
          :return:
          """
          from returnn.tf.compat import v1 as tf
          model_d = tf.cast(model_d, tf.float32)
          n = tf.cast(n, tf.float32)
          warmup_n = tf.cast(warmup_n, tf.float32)
          return tf.pow(model_d, -0.5) * tf.minimum(tf.pow(n, -0.5), n * tf.pow(warmup_n, -1.5))

        def dynamic_learning_rate(*, network, global_train_step, learning_rate, **kwargs):
          """

          :param TFNetwork network:
          :param tf.Tensor global_train_step:
          :param tf.Tensor learning_rate: current global learning rate
          :param kwargs:
          :return:
          """
          WARMUP_N = 25000
          MODEL_D = 512
          return learning_rate * noam(n=global_train_step, warmup_n=WARMUP_N, model_d=MODEL_D)


gradient_clip
    Specify a gradient clipping threshold.

gradient_noise
    Apply a (gaussian?) noise to the gradient with given deviation (variance? stddev?)

learning_rate
    Specifies the global learning rate

learning_rates
    A list of learning rates that defines the learning rate for each epoch from the beginning.
    Can be used for learning-rate warmup.

learning_rate_control
    This defines which type of learning rate control mechanism is used. Possible values are:

    - ``constant`` for a constant learning rate which is never modified
    - ``newbob_abs`` for a scheduling based on absolute improvement
    - ``newbob_rel`` for a scheduling based on relative improvement
    - ``newbob_multi_epoch`` for a scheduling based on relative improvement averaged over multiple epochs

    Please also look at setting values with the ``newbob`` prefix for further customization

learning_rate_control_error_measure
    A str to define which score or error is used to control the learning rate reduction.
    Per default, Returnn will use dev_score_output.
    A typical choice would be dev_score_LAYERNAME or dev_error_LAYERNAME.
    Can be set to None to disable learning rate control.

learning_rate_control_min_num_epochs_per_new_lr
    The number of epochs after the last update that the learning rate is kept constant.

learning_rate_control_relative_error_relative_lr
    If true, the relative error is scaled with the ratio of the default learning rate divided by the current
    learning rate.
    Can be used with ``newbob_rel`` and ``newbob_multi_epoch``.

learning_rate_file
    A path to a file storing the learning rate for each epoch.
    Despite the name, also stores scores and errors.

min_learning_rate
    Specifies the minimum learning rate.

newbob_error_threshold
    This is the absolute improvement that has to be achieved in order to _not_ reduce the learning rate.
    Can be used with ``newbob_abs``.
    The value can be positive or negative.

newbob_learning_rate_decay
    The scaling factor for the learning rate when a reduction is applied.
    This parameter is available for all ``newbob`` variants.

newbob_multi_num_epochs
    The number of epochs the improvement is averaged over.

newbob_multi_update_interval
    The number of steps after which the learning rate is updated.
    This is set equal to ``newbob_multi_num_epochs`` when not specified.

newbob_relative_error_threshold
    This is the relative improvement that has to be achieved in order to _not_ reduce the learning rate.
    Can be used with ``newbob_rel`` and ``newbob_multi_epoch``.
    The value can be positive or negative.

optimizer
    A dictionary with a ``class`` entry for the optimizer.
    Other keys are passed as parameters to the constructor of the optimizer class.

relative_error_div_by_old
    If true the relative error is computed by dividing the error difference by the old error value instead of the
    current error value.

reset_updater_vars_mod_step
    The number of epochs after which the internal states of all optimizers will be resetted to their initial state.

use_learning_rate_control_always
    If true, use the learning rate control scheme also during pre-training.
