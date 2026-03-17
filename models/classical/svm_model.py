from sklearn.svm import SVC


def create_svm():

    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True
    )

    return model