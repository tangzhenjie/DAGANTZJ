# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np


class RunningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class**2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes)

    def per_class_iu(self, hist):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        mIoUs = self.per_class_iu(hist)
        print('===> mIoU19: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        print('===> mIoU16: ' + str(
            round(np.mean(mIoUs[[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]]) * 100, 2)))
        print('===> mIoU13: ' + str(round(np.mean(mIoUs[[0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]]) * 100, 2)))

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))