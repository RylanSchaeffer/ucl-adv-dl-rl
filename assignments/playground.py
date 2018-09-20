import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.contrib.distributions import RelaxedOneHotCategorical
from tensorflow.python.framework import ops
from tensorflow.python.layers import base
from tensorflow.python.ops import init_ops


tfe.enable_eager_execution()


class LearnableDropoutDense(base.Layer):

    def init(self, units,
             activation=None,
             use_bias=True,
             kernel_initializer=None,
             bias_initializer=init_ops.zeros_initializer(),
             kernel_regularizer=None,
             bias_regularizer=None,
             activity_regularizer=None,
             kernel_constraint=None,
             bias_constraint=None,
             trainable=True,
             name=None,
             **kwargs):

        self.units = units
        self.activation = activation

        super(LearnableDropoutDense, self).__init__(name=name, **kwargs)
        self.kernel = self.add_variable('kernel',
                                        shape=[input_shape[-1].value, self.units],
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        dtype=self.dtype,
                                        trainable=True)

    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        outputs = standard_ops.matmul(inputs, self.output_kernel)




temperature = 1e-5
p = [0.1, 0.8, 0.1]
dist = RelaxedOneHotCategorical(temperature=temperature, probs=p)
r = dist.sample(sample_shape=1000)
print(r)
