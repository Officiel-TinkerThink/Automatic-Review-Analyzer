"""pytest suite for project1.py (the original NumPy implementation).  Run:  pytest -q"""
import numpy as np

import project1 as p1


def test_hinge_loss():
    assert np.isclose(p1.hinge_loss_single(np.array([1.0, 2.0]), 1, np.array([-1.0, 1.0]), -0.2), 0.2)
    fm = np.array([[1.0, 2.0], [1.0, 2.0]])
    assert np.isclose(p1.hinge_loss_full(fm, np.array([1, 1]), np.array([-1.0, 1.0]), -0.2), 0.2)


def test_perceptron_single_step():
    theta, theta0 = p1.perceptron_single_step_update(np.array([1.0, 2.0]), 1, np.array([-1.0, 1.0]), -1.5)
    assert np.allclose(theta, [0.0, 3.0]) and theta0 == -0.5
    theta, theta0 = p1.perceptron_single_step_update(np.array([1.0, 2.0]), 1, np.array([1.0, 1.0]), 1.0)
    assert np.allclose(theta, [1.0, 1.0]) and theta0 == 1.0  # already correct → no update


def test_pegasos_single_step():
    theta, theta0 = p1.pegasos_single_step_update(np.array([1.0, 2.0]), 1, 0.2, 0.1, np.array([-1.0, 1.0]), -1.5)
    assert np.allclose(theta, [-0.88, 1.18]) and abs(theta0 - (-1.4)) < 1e-9


def test_algorithms_separate_toy_data():
    fm = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    labels = np.array([1, -1, 1, -1])
    for fn, kw in [(p1.perceptron, {"T": 5}), (p1.average_perceptron, {"T": 5}), (p1.pegasos, {"T": 5, "L": 0.01})]:
        theta, theta0 = fn(fm, labels, **kw)
        assert (p1.classify(fm, theta, theta0) == labels).all()


def test_bag_of_words_and_features():
    d = p1.bag_of_words(["Good, but bad", "good"])
    assert d == {"good": 0, ",": 1, "but": 2, "bad": 3}
    fm = p1.extract_bow_feature_vectors(["good good bad"], d)
    assert fm.tolist() == [[1, 0, 0, 1]]
    fm = p1.extract_bow_feature_vectors(["good good bad"], d, binarize=False)
    assert fm.tolist() == [[2, 0, 0, 1]]


def test_reaches_course_accuracy_on_real_reviews():
    import utils
    train = utils.load_data("reviews_train.tsv")
    val = utils.load_data("reviews_val.tsv")
    tx, ty = zip(*((s["text"], s["sentiment"]) for s in train))
    vx, vy = zip(*((s["text"], s["sentiment"]) for s in val))
    d = p1.bag_of_words(tx)
    X, XV = p1.extract_bow_feature_vectors(tx, d), p1.extract_bow_feature_vectors(vx, d)
    theta, theta0 = p1.average_perceptron(X, np.array(ty), T=10)
    acc = p1.accuracy(p1.classify(XV, theta, theta0), np.array(vy))
    assert acc > 0.75
