class StereoRegression():
    def __init__(self, unary, cost, regression, classification, loss):
        self.unary = unary
        self.cost = cost
        self.regression = regression
        self.classification = classification
        self.loss = loss
        self.unary_outputs = {}
        self.cost_volumes = {}
        self.regressions = {}
        self.outputs = {}
        self.losses = {}

    def build(self, placeholders, is_training, reuse):
        self.unary.build(placeholders, is_training)
        unary_outputs = self.unary.embeddings[placeholders]
        self.cost.build(unary_outputs)
        cost_volume = self.cost.volumes[unary_outputs]
        self.regression.build(cost_volume, is_training, reuse)
        projection = self.regression.projections[cost_volume]
        self.classification.build(projection)
        output = self.classification.outputs[projection]
        self.loss.build(output, placeholders)
        loss = self.loss.losses[output]

        self.unary_outputs[placeholders] = unary_outputs
        self.cost_volumes[placeholders] = cost_volume
        self.regressions[placeholders] = projection
        self.outputs[placeholders] = output
        self.losses[placeholders] = loss