import copy, warnings, sklearn, inspect
from scipy.interpolate import RBFInterpolator
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from . import geometry


def getWeightsForNonUniformSample(x):
    """
    Calculates weights for each object x[i] = NN_dist^dim. These weights make uniform the error of ML models fitted on non-uniform samples
    """
    assert len(x.shape) == 2
    if len(x) <= 1: return np.ones(len(x))
    NNdists, _ = geometry.getNNdistsStable(x, x)
    w = NNdists**x.shape[1]
    w /= np.sum(w)
    w[w<1e-6] = 1e-6
    w /= np.sum(w)
    return w


def crossValidation(estimator, X, Y, CVcount, YColumnWeights=None, nonUniformSample=False):
    if isinstance(X, pd.DataFrame): X = X.to_numpy()
    if isinstance(Y, pd.DataFrame): Y = Y.to_numpy()
    if isinstance(Y, list): Y = np.array(Y)
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)
    if len(Y.shape) == 1: Y = Y.reshape(-1, 1)
    N = Y.shape[0]
    assert len(X) == N
    if YColumnWeights is None:
        YColumnWeights = np.ones((1,Y.shape[1]))
    if N > 20:
        kf = sklearn.model_selection.KFold(n_splits=CVcount, shuffle=True, random_state=0)
    else:
        kf = sklearn.model_selection.LeaveOneOut()
    predictedY = np.zeros(Y.shape)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index,:], X[test_index,:]
        y_train, y_test = Y[train_index, :], Y[test_index, :]
        estimator.fit(X_train, y_train)
        predictedY[test_index] = estimator.predict(X_test)
    if nonUniformSample: rowWeights = getWeightsForNonUniformSample(X)
    else: rowWeights = np.ones(N)
    individualErrors = np.array([np.sqrt(np.sum(np.abs(Y[i] - predictedY[i])**2 * YColumnWeights)) for i in range(N)])
    u = np.sum(individualErrors*rowWeights)
    y_mean = np.mean(Y, axis=0)
    v = np.sum(np.array([np.sqrt(np.sum(np.abs(Y[i] - y_mean)**2 * YColumnWeights)) for i in range(N)]) * rowWeights)
    error = u / v
    return error, individualErrors, predictedY


class RBF:
    def __init__(self, function='linear', baseRegression='quadric', scaleX=True, removeDublicates=False):
        """
        RBF predictor
        :param function: string. Possible values: multiquadric, inverse, gaussian, linear, cubic, quintic, thin_plate
        :param baseRegression: string, base estimator. Possible values: quadric, linear, None
        :param scaleX: bool. Scale X by gradients of y
        """
        # function: multiquadric, inverse, gaussian, linear, cubic, quintic, thin_plate
        # baseRegression: linear quadric
        self.function = function
        self.baseRegression = baseRegression
        self.trained = False
        self.scaleX = scaleX
        self.train_x = None
        self.train_y = None
        self.base = None
        self.scaleGrad = None
        self.minX = None
        self.maxX = None
        self.interp = None
        self.removeDublicates = removeDublicates

    def get_params(self, deep=True):
        return {'function': self.function, 'baseRegression': self.baseRegression, 'scaleX': self.scaleX}

    def set_params(self, **params):
        self.function = copy.deepcopy(params['function'])
        self.baseRegression = copy.deepcopy(params['baseRegression'])
        self.scaleX = copy.deepcopy(params['scaleX'])
        return self

    def fit(self, x, y):
        x = copy.deepcopy(x)
        y = copy.deepcopy(y)
        self.train_x = x.values if (type(x) is pd.DataFrame) or (type(x) is pd.Series) else x
        self.train_y = y.values if (type(y) is pd.DataFrame) or (type(y) is pd.Series) else y
        if len(self.train_y.shape) == 1: self.train_y = self.train_y.reshape(-1, 1)
        if self.baseRegression == 'quadric': self.base = makeQuadric(RidgeCV())
        elif self.baseRegression is None: self.base = None
        else:
            assert self.baseRegression == 'linear'
            self.base = RidgeCV()
        if self.scaleX:
            n = self.train_x.shape[1]
            self.minX = np.min(self.train_x, axis=0)
            self.maxX = np.max(self.train_x, axis=0)
            self.train_x = norm(self.train_x, self.minX, self.maxX)
            quadric = makeQuadric(RidgeCV())
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                quadric.fit(self.train_x, self.train_y)
            center_x = np.zeros(n)
            center_y = quadric.predict(center_x.reshape(1,-1))
            grad = np.zeros(n)
            for i in range(n):
                h = 1
                x2 = np.copy(center_x)
                x2[i] = center_x[i] + h
                y2 = quadric.predict(x2.reshape(1,-1))
                x1 = np.copy(center_x)
                x1[i] = center_x[i] - h
                y1 = quadric.predict(x1.reshape(1,-1))
                grad[i] = np.max([np.linalg.norm(y2 - center_y, ord=np.inf) / h, np.linalg.norm(center_y - y1, ord=np.inf) / h])
            if np.max(grad) == 0:
                if self.train_x.shape[0] > 2:
                    warnings.warn(f'Constant function. Gradient = 0. x.shape={self.train_x.shape}')
                self.scaleGrad = np.ones((1,n))
            else:
                grad = grad / np.max(grad)
                eps = 0.01
                if len(grad[grad <= eps]) > 0:
                    grad[grad <= eps] = np.min(grad[grad > eps]) * 0.01
                self.scaleGrad = grad.reshape(1,-1)
                self.train_x = self.train_x * self.scaleGrad
        if self.removeDublicates:
            # RBF crashes when dataset includes close or equal points
            self.train_x, uniq_ind = geometry.unique_mulitdim(self.train_x)
            self.train_y = self.train_y[uniq_ind,:]
        w = getWeightsForNonUniformSample(self.train_x)
        if self.baseRegression is not None:
            with warnings.catch_warnings():
                # warnings.simplefilter("ignore")
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                self.base.fit(self.train_x, self.train_y, sample_weight=w)
            self.train_y = self.train_y - self.base.predict(self.train_x)
        NdimsY = self.train_y.shape[1]
        assert NdimsY > 0
        self.interp = RBFInterpolator(self.train_x, self.train_y, kernel=self.function, degree=0)
        self.trained = True

    def predict(self, x):
        assert self.trained
        if type(x) is pd.DataFrame: x = x.values
        assert len(x.shape) == 2, f'x = '+str(x)
        assert x.shape[1] == self.train_x.shape[1], f'{x.shape[1]} != {self.train_x.shape[1]}'
        if self.scaleX:
            x = norm(x, self.minX, self.maxX)
            x = x * self.scaleGrad
        res = self.interp(x)
        if self.baseRegression is not None:
            res = res + self.base.predict(x)
        return res


def transformFeatures2Quadric(x, addConst=True):
    isDataframe = type(x) is pd.DataFrame
    if isDataframe:
        col_names = np.array(x.columns)
        x = x.values
    n = x.shape[1]
    new_n = n + n*(n+1)//2
    if addConst: new_n += 1
    newX = np.zeros([x.shape[0], new_n])
    newX[:,:n] = x
    if isDataframe:
        new_col_names = np.array(['']*newX.shape[1], dtype=object)
        new_col_names[:n] = col_names
    k = n
    for i1 in range(n):
        for i2 in range(i1,n):
            newX[:,k] = x[:,i1]*x[:,i2]
            if isDataframe:
                if i1 != i2:
                    new_col_names[k] = col_names[i1]+'*'+col_names[i2]
                else:
                    new_col_names[k] = col_names[i1] + '^2'
            k += 1
    if addConst:
        newX[:,k] = 1
        if isDataframe:
            new_col_names[k] = 'const'
        k += 1
        assert k == n + n*(n+1)//2 + 1
    else: assert k == n + n*(n+1)//2
    if isDataframe:
        newX = pd.DataFrame(newX, columns=new_col_names)
    return newX


class makeQuadric:
    def __init__(self, learner):
        self.learner = learner

    def get_params(self, deep=True):
        return {'learner': self.learner}

    def set_params(self, **params):
        self.learner = copy.deepcopy(params['learner'])
        return self

    def fit(self, x, y, **args):
        x2 = transformFeatures2Quadric(x)
        self.learner.fit(x2, y, **args)

    def predict(self, x):
        return self.learner.predict(transformFeatures2Quadric(x))


def norm(x, minX, maxX):
    """
    Do not norm columns in x for which minX == maxX
    :param x:
    :param minX:
    :param maxX:
    :return:
    """
    dx = maxX-minX
    ind = dx != 0
    res = copy.deepcopy(x)
    if type(x) is pd.DataFrame:
        res.loc[:, ind] = 2 * (x.loc[:, ind] - minX[ind]) / dx[ind] - 1
        res.loc[:,~ind] = 0
    else:
        if minX.size == 1:
            if dx != 0: res = 2 * (x - minX) / dx - 1
            else: res[:] = 0
        else:
            res[:, ind] = 2 * (x[:, ind] - minX[ind]) / dx[ind] - 1
            res[:, ~ind] = 0
    return res


def invNorm(x, minX, maxX):
    """
    Do not norm columns in x for which minX == maxX
    :param x:
    :param minX:
    :param maxX:
    :return:
    """
    dx = maxX - minX
    ind = dx != 0
    res = copy.deepcopy(x)
    if type(x) is pd.DataFrame:
        res.loc[:, ind] = (x.loc[:, ind]+1)/2*dx[ind] + minX[ind]
        res.loc[:, ~ind] = minX[~ind]
    else:
        if minX.size == 1:
            if dx != 0: res = (x+1)/2*(maxX-minX) + minX
            else: res[:] = minX
        else:
            res[:, ind] = (x[:, ind]+1)/2*dx[ind] + minX[ind]
            res[:, ~ind] = minX[~ind]
    return res


class Normalize:
    def __init__(self, learner, xOnly):
        self.learner = learner
        self.xOnly = xOnly
        if not hasattr(learner, 'name'): self.name = str(type(learner))
        else: self.name = 'normalized '+learner.name

    def get_params(self, deep=True):
        return {'learner':self.learner, 'xOnly':self.xOnly}

    def set_params(self, **params):
        self.learner = copy.deepcopy(params['learner'])
        self.xOnly= params['xOnly']
        return self

    def isFitted(self):
        return isFitted(self.learner)

    # args['xyRanges'] = {'minX':..., 'maxX':..., ...}
    def fit(self, x, y, **args):
        if isinstance(y,np.ndarray) and (len(y.shape)==1): y = y.reshape(-1,1)
        y_is_df = type(y) is pd.DataFrame
        if y_is_df: columns = y.columns
        if 'xyRanges' in args: self.xyRanges = args['xyRanges']; del args['xyRanges']
        else: self.xyRanges = {}
        if len(self.xyRanges)>=2:
            self.minX = self.xyRanges['minX']; self.maxX = self.xyRanges['maxX']
            if len(self.xyRanges)==4: self.minY = self.xyRanges['minY']; self.maxY = self.xyRanges['maxY']
        else:
            self.minX = np.min(x, axis=0); self.maxX = np.max(x, axis=0)
            if self.xOnly:
                self.minY = -np.ones(y.shape[1])
                self.maxY = np.ones(y.shape[1])
            else:
                self.minY = np.min(y.values, axis=0) if y_is_df else np.min(y, axis=0)
                self.maxY = np.max(y.values, axis=0) if y_is_df else np.max(y, axis=0)
        if type(self.minX) is pd.Series: self.minX = self.minX.values; self.maxX = self.maxX.values
        if type(self.minY) is pd.Series: self.minY = self.minY.values; self.maxY = self.maxY.values
        # print(self.minX, self.maxX, self.minY, self.maxY)
        if 'validation_data' in args:
            (xv, yv) = args['validation_data']
            validation_data = (norm(xv, self.minX, self.maxX), norm(yv, self.minY, self.minY))
            args['validation_data'] = validation_data
        if 'yRange' in args: args['yRange'] = [norm(args['yRange'][0], self.minY, self.minY), norm(args['yRange'][1], self.minY, self.minY)]
        self.learner.fit(norm(x, self.minX, self.maxX), norm(y, self.minY, self.maxY), **args)
        return self

    def predict(self, x, **predictArgs):
        if type(x) is pd.DataFrame: x = x.values
        yn = self.learner.predict(norm(x, self.minX, self.maxX), **predictArgs)
        if isinstance(yn, tuple):
            return (invNorm(yn[0], self.minY, self.maxY),) + yn[1:]
        else:
            return invNorm(yn,self.minY, self.maxY)

    def predict_proba(self, x):
        pyn = self.learner.predict_proba(norm(x, self.minX, self.maxX))
        return pyn


def isFitted(estimator):
    if isinstance(estimator, sklearn.base.BaseEstimator):
        try:
            sklearn.utils.validation.check_is_fitted(estimator)
        except sklearn.exceptions.NotFittedError:
            return False
        return True
    if hasattr(estimator, 'isFitted') and callable(getattr(estimator, 'isFitted')):
        return estimator.isFitted()
    if hasattr(estimator, "classes_"): return True
    if 0 < len( [k for k,v in inspect.getmembers(estimator) if k.endswith('_') and not k.startswith('__')] ): return True
    assert hasattr(estimator, 'trained'), 'Your estimator is very unusual. Use your custom isFitted method'
    return estimator.trained
