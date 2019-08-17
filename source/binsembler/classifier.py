from __future__ import division
from binsembler.utils.common import *
from binsembler.utils.log import *
from binsembler.utils.timer import Timer
from functools import reduce
from binsembler.helpers.bins import *
from binsembler.helpers.classification_evaluator import *

setup_logger(log_level="INFO")  # TODO: Make it configurable as a start up param


class Classifier:
    """
    Creates an ensemble model using the models input to the function
    1.  Based on the predicted probabilities, create bins of fixed size (say 10)
    2.  For each bin, calculate the Confusion matrix (TP, FP, TN, FN) and calculate other metrics such as Accuracy, AUC, F1 score, Precision, Recall
    3.  Repeat the steps 1 and 2 for other model
    4.  Calculate the final probability as a weighted average of models probabilities
        Select any metric as weights. For e.g. F1 Score
        (m1 F1 score * m1 predicted probability + m2 F1 score * m2 predicted probability)/ (m1 F1 score + m2 F1 score)
    """

    def __init__(self, nbins=10, metric="Accuracy", models=[], targetcols=[]):
        """
        Initializes the class level variables
        :param models:
        :param targetCols:
        :param nbins:
        :param metric:
        """
        self._nbins = nbins
        self._models = models
        self._targetcols = targetcols
        self.w_ = pd.DataFrame()
        self.threshold = 0.5
        self._metric = metric
        self.pred_df = pd.DataFrame()

    @Timer(logging.getLogger())
    def train(self, models=[], train_data=None, targetcols=[], X_train=[], y_train=[], y_pred_probs=[], y_acts=[], y_preds=[], thresholds=[], metric="Accuracy"):
        """
        Here a model is nothing but binwise metric (Accuracy, F1, AUC etc.) calculated based on the probabilities provided
        :param models:
        :param train_data:
        :param targetcols:
        :param X_train:
        :param y_train:
        :param y_pred_probs:
        :param y_acts:
        :param y_preds:
        :param thresholds:
        :param metric:
        :return:
        """

        self._models = models
        self._targetcols = targetcols
        if self._metric is None:
            self._metric = metric

        # validations
        is_model_given = False
        is_prob_given = False

        if len(models) > 0: # and len(targetcols) > 0:
            is_model_given = True

        if len(y_pred_probs) >0 and len(y_acts) > 0:
            is_prob_given = True

        if not is_model_given and not is_prob_given:
            assert False, "Either models or predicted probability by models must be provided"

        if is_prob_given:
            n = len(y_pred_probs)

            if n != len(y_acts):
                assert False, "Predicted Probabilities and actual values must be of same length"

        if is_model_given:
            n = len(models)
            if train_data is None and len(X_train) == 0:
                assert False, "train_data or X_train must be provided to train the model"
            if train_data is not None:
                if len(models) != len(targetcols):
                    targetcols = np.reshape(targetcols, (1, len(models)))
            else:
                if len(X_train) != len(y_train):
                    assert False, "X_train or y_train must have same length"

        # Loop through each model's prediction probabilities and actual values to calculate the binwise stats
        model_metrics = []

        for i in range(0, n):

            if is_model_given:
                model = models[i]
                if train_data is not None:
                    targetcol = targetcols[i]
                    y_act = train_data[targetcol]
                    if is_h2o_frame(y_act):
                        y_act = convert_h2o_list(y_act)
                    y_pred, y_pred_prob = self._get_model_preds(model, train_data)
                else:
                    y_act = y_train
                    y_pred, y_pred_prob = self._get_model_preds(model, X_train)

                y_pred_probs.append(y_pred_prob)
            else:
                y_pred_prob = y_pred_probs[i]
                y_act = y_acts[i]
                y_pred = []
                if len(y_preds) > 0:
                    if len(y_preds[i]) > 0:
                        y_pred = y_preds[i]
                if len(thresholds) == 0:
                    threshold = 0.5
                else:
                    threshold = thresholds[i]

                # do the validation inside the each model to see predicted probabilities and actual values are provided
                if len(y_pred_prob) == 0:
                    assert False, "Predicted Probabilities must be provided to ensemble"

                if len(y_act) == 0:
                    assert False, "y actual must be provided to ensemble"

                if len(y_pred_prob) != len(y_act):
                    assert False, "Predicted Probabilities and actual values must be of same length"

                # check if preds are given, if yes attach this or derive based on threshold
                if len(y_pred) == 0:
                    y_pred = self._get_prob_preds(y_pred_prob, threshold)

            # Get probability the bins
            prob_bins = get_bins(y_pred_prob, self._nbins)
            # Get the confusion matrix for each bin for p1
            cm_df = get_cm(y_pred, y_act, prob_bins['bins'].values.tolist())
            # Calculate all the metrics
            metrics_df = get_metrics(cm_df)
            metrics_df = metrics_df.add_prefix('m' + str(i + 1) + '_')
            model_metrics.append(metrics_df)

        bin_metrics = reduce(lambda left, right: left.join(right), model_metrics).fillna(0) #pd.merge(left, right) #pd.concat(model_metrics)

        # bin metrics are our model coefficients
        self.w_ = bin_metrics

        # get the threshold for best F1 or Accuracy
        self.threshold = self._get_threshold(self._metric, pred_probs=y_pred_probs)

        return self

    def _get_threshold(self, metric, pred_probs):
        """
        Gets the threshold for best F1 or a given metric. Only F1 is supported now
        The best threshold for the ensemble model is arrived at making predictions on the whole training set
        :param pred_probs:
        :param acts:
        :param metric:
        :return:
        """
        preds_df = self.predict(y_pred_probs = pred_probs, metric = metric)
        # finds optimal threshold. Let's pick any of the actuals
        threshold = get_optimal_threshold(preds_df['ensmbl_preds'].values,
                                          preds_df['ensmbl_pred_probs'].values)
        return threshold

    def _get_prob_preds(self, y_pred_probs, threshold):
        """
        Makes predictions using probabilities and threshold
        :param y_pred_probs:
        :param threshold: 
        :return: 
        """
        y_preds = np.where(np.asarray(y_pred_probs) >= threshold, 1, 0)
        y_preds = y_preds.tolist()

        return y_preds

    def _get_model_preds(self, model, data):
        """
        Makes predictions using the given model and data
        :param model:
        :param data:
        :param targetcol:
        :return:
        """
        msg = ""
        try:
            preds = model.predict(data)
            if preds.ndim == 1:
                y_preds = preds
                # get the probabilities in case if it is a different function call
                probs = model.predict_proba(data)
                y_pred_probs = probs[:, 0]
                y_pred_probs0 = probs[:, 1]
            else:
                y_preds = preds[:, 0]
                y_pred_probs = y_preds[:, 2]
                y_pred_probs0 = y_preds[:, 1]
        except Exception as e:
            msg = "Model must expose predict() or predict_proba() method"
            logging.error(msg)
            assert False, msg

        # make necessary transformation for h20 frames
        if is_h2o_frame(y_preds):
            y_preds = convert_h2o_list(y_preds)

        if is_h2o_frame(y_pred_probs):
            y_pred_probs = convert_h2o_list(y_pred_probs)

        if is_h2o_frame(y_pred_probs0):
            y_pred_probs0 = convert_h2o_list(y_pred_probs0)

        return y_preds, y_pred_probs

    @Timer(logging.getLogger())
    def predict(self, models=[], test_data=None, y_pred_probs=[], metric="Accuracy"):
        """
        Combines the predicted probabilities from any number of models as weighted average where weights are the binwise
        metric specified
        :return:
        """
        # validations
        is_model_given = False
        is_prob_given = False

        if len(models) > 0:
            is_model_given = True

        if len(y_pred_probs) > 0:
            is_prob_given = True

        if not is_model_given and not is_prob_given:
            assert False, "Either models or predicted probability by models must be provided"

        if is_prob_given:
            n = len(y_pred_probs)

        if is_model_given:
            n = len(models)
            if test_data is None:
                assert False, "Test data must be provided to train the model"

        # make final prediction
        preds_df = pd.DataFrame()
        numer = []  # numerator of ensemble formula
        denom = []  # denominator of ensemble formula

        for i in range(0, n):
            if is_model_given:
                model = models[i]
                y_pred, y_pred_prob = self._get_model_preds(model, test_data)
            else:
                y_pred_prob = y_pred_probs[i]

            # first get the bins for the new prediction probabilities
            prob_bins = get_bins(y_pred_prob, self._nbins)
            prob_bins = prob_bins.add_prefix('test' + str(i + 1) + '_')

            prob_col_name = 'y' + str(i + 1) + '_pred_probs'
            join_col_r = 'm' + str(i + 1) + '_bins'
            join_col_l = 'test' + str(i + 1) + '_bins'
            preds_df[prob_col_name] = y_pred_prob

            preds_df = preds_df.join(prob_bins)
            pattern = 'm' + str(i + 1) + '_.*'
            preds_df = pd.merge(preds_df, self.w_.filter(regex=(pattern)), left_on=join_col_l,
                                right_on=join_col_r,
                                how='left')

            metricl = 'm' + str(i + 1) + '_' + self._metric

            # replacing na values with 0
            preds_df[metricl].fillna(0, inplace=True)
            # check if any of these metric values is zero, if yes, give a smaller value instead of zero, e.g. 0.001 to make
            # sure the the model effect is not nullified at all
            preds_df.loc[preds_df[metricl] == 0, metricl] = 0.001
            part_numer = preds_df[prob_col_name] * preds_df[metricl]
            # numer = numer + part_numer
            # denom = denom + preds_df[metricl]
            numer.append(part_numer)
            denom.append(preds_df[metricl])

        numerl = reduce(lambda left, right: left + right, numer)
        denoml = reduce(lambda left, right: left + right, denom)

        # Ensembling two models prediction
        preds_df['ensmbl_pred_probs'] = numerl / denoml
        preds_df['ensmbl_pred_probs0'] = 1 - preds_df['ensmbl_pred_probs']

        # predict based on threshold
        preds_df['ensmbl_preds'] = np.where(preds_df['ensmbl_pred_probs'] >= self.threshold, 1, 0)
        # ensmeble bins
        bins_df = get_bins(preds_df['ensmbl_pred_probs'].values, self._nbins)
        preds_df = preds_df.join(bins_df, rsuffix="pred_bin")
        #self.w_ = self.w_.join(bins_df, rsuffix="_pred_bin")
        return preds_df[['ensmbl_preds', 'ensmbl_pred_probs', 'ensmbl_pred_probs0']]

    def get_threshold(self):
        """
        Returs the threshold found for best F1 on the validation data set while training the ensemble
        :return:
        """
        return self.threshold
