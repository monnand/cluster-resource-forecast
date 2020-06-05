from predictor import StatefulPredictor
import numpy as np


class MaxDecorator(StatefulPredictor):
    def __init__(self, config, decorated_predictors=None):
        self.config = config
        self.decorated_predictors = decorated_predictors

    def CreateState(self, vm_info):
        pass

    def UpdateMeasures(self, snapshot):
        predictions = []
        for predictor in self.decorated_predictors:
            predictions.append(predictor.UpdateMeasures(snapshot))

        return self.Predict(predictions)

    def UpdateState(self, vm_measure, vm_state):
        pass

    def Predict(self, predictions):

        limits = []
        predicted_peaks = []
        for prediction_limit in predictions:
            limits.append(prediction_limit[0])
            predicted_peaks.append(prediction_limit[1])

        return (np.max(limits), np.max(predicted_peaks))
