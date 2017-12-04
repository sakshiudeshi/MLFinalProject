from pyfm import pylibfm
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction import DictVectorizer
import model
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
# from polylearn import PolynomialNetworkRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor


from math import sqrt

regr = LinearSVR(random_state=0)
svr_lin = SVR(kernel='linear', C=1e3)
fm = pylibfm.FM()
knr = KNeighborsRegressor(n_neighbors=10)
dr = DummyRegressor()
bagrgr = BaggingRegressor()
dtreergr = DecisionTreeRegressor()
adabregr = AdaBoostRegressor()
gradbregr = GradientBoostingRegressor()

def validate(X, y):
    print "Starting cross validation"
    scores = cross_val_score(knr, X, y, scoring='neg_mean_squared_error', cv=3)
    return scores


if __name__ == "__main__":
    import music

    train_examples = music.load_examples('data/train.pkl')
    # poly = PolynomialNetworkRegressor(degreex=3, n_components=2, tol=1e-3, warm_start=True, random_state=0)
    fm = pylibfm.FM(num_iter=10, verbose=True, task="regression", initial_learning_rate=0.001, learning_rate_schedule="optimal")
    v = DictVectorizer()
    X = np.asarray([model.represent(example) for example in train_examples])
    y = np.asarray([model.label(example) for example in train_examples])
    # fm.fit(sparse.csr_matrix(X), y)
    # svr_rbf.fit(X, y)
    # pca = PCA(n_components=100)
    # pca.fit(X)
    # X_fit = pca.transform(X)
    # print "pca done"
    # xs = [x[0] for x in X_fit]
    # ys = [x[1] for x in X_fit]
    # plt.scatter(xs, ys)
    # plt.show()
    # print v.fit_transform(X)
    # print X_fitM
    y_np = np.asarray(y)
    plt.hist(y_np, bins=np.arange(y_np.min(), y_np.max() + 1))
    # plt.title("Frequency of ratings")
    # plt.show()

    # fm.fit(csr_matrix(X), y)
    scores = validate(X, y)

    print "GradientBoosting regressor: %0.6f (+/- %0.6f)"    % (sqrt(-1 * scores.mean()), scores.std() / 2)
