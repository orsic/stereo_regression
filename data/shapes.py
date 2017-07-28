class InputShape():
    def __init__(self, width, height, channels, max_disp, semantic_features):
        self.W = width
        self.H = height
        self.C = channels
        self.D = max_disp
        self.S = semantic_features
        self.image_shape = (1, height, width, channels)
        self.disparity_shape = (1, height, width, 1)
        self.semantic_shape = (1, height // 4, width // 4, semantic_features)
        self.label_shape = (1, height, width, 1)
        self.softmax_shape = (1, height, width, 19)