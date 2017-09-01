"""Python implementation of RankingLossLayer."""
import caffe
import numpy as np
import os
import sys

class RankingLossLayer(caffe.Layer):

    def setup(self, bottom, top):

        r'''Setup the layer with params.
        Example params:
        layer {
          name: "loss"
          type: "Python"
          bottom: "feat"
          bottom: "feat_p"
          top: "loss"
          loss_weight: 1
          python_param {
            module: "pyRankLoss"
            layer: "RankingLossLayer"
            param_str: "{\'loss_weight\': 1, ", "\'normalization\': 1}"
          }
        }
        '''

        #Check bottom shape
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

        #Read parameters
        params = eval(self.param_str)
        self._loss_weight = params.get('loss_weight', 1)
        self._normalization = params.get('normalization', 2)


    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        if bottom[1].channels != 1:
            raise Exception("Inputs must be scalar.")   
        # difference is shape of inputs
        self.prob = np.zeros(bottom[0].num, dtype=np.float32)   
        self.ones = np.ones(bottom[0].num)
        # loss output is scalar
        top[0].reshape(1)


    def forward(self, bottom, top):
        diff = bottom[0].data - bottom[1].data
        loss = self.ones / (self.ones + np.exp(diff))
        self.prob = loss
        top[0].data[...] = loss * self._loss_weight / float(self.get_normalizer(bottom[0].data))


    def backward(self, top, propagate_down, bottom):
        for i, sign in enumerate([ +1, -1 ]):
            if propagate_down[i]:
                print "bottom ", i, " is ", bottom[i].data.shape
                bottom_diff = self.prob ** 2 - self.prob
                loss_weight = sign * self._loss_weight / float(self.get_normalizer(bottom[0].data))
                bottom[i].diff[...] = bottom_diff * loss_weight


    def get_normalizer(self, scores):
        """Get the loss normalizer based normalization mode."""
        if self._normalization == 0:    # Full
            normalizer = scores.size
        elif self._normalization == 1:  # VALID
            normalizer = scores.size
        elif self._normalization == 2:  # BATCH_SIZE
            normalizer = scores.shape[0]
        elif self._normalization == 3:  # NONE
            normalizer = 1.
        else:
            raise Exception("Unknown normalization mode: {}").format(
                self._normalization)
        return max(1., normalizer)


