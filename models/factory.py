from models import unary, cost_volume, regression, classification, loss, stereo_regression

unaries = {
    'resnet': unary.Resnet,
}

cost_volumes = {
    'concat': cost_volume.CostVolume,
}

regressions = {
    'resnet': regression.ResnetRegression,
    'resnetcat19': regression.ResnetRegressionConcat,
}

classifications = {
    'softargmin': classification.SoftArgmin,
}

losses = {
    'abs': loss.Loss,
    'unsupervised': loss.UnsupervisedLoss,
    'unsupervised_ssim': loss.UnsupervisedLossSSIM,
}


def create_model(flags, config):
    uf = unaries[config.unary](flags, config)
    cv = cost_volumes[config.cost_volume](flags, config)
    dr = regressions[config.regression](flags, config)
    cl = classifications[config.classification](flags, config)
    ls = losses[config.loss](flags, config)
    model = stereo_regression.StereoRegression(uf, cv, dr, cl, ls)
    return model
