from collections import namedtuple

DatasetSubsets = namedtuple('Generators', 'examples name')

class Dataset():
    def __init__(self, example_class, paths, name):
        self.name = name
        self.paths = paths
        self.example_class = example_class

    def __getattr__(self, item):
        assert item in self.paths
        for key in ['left', 'right']:
            if len(self.paths[item]) > 0:
                assert key in self.paths[item][0]
        return self.generator(item)

    def generator(self, subset):
        return (self.example_class(path) for path in self.paths[subset])

    def __iter__(self):
        for name in self.paths:
            yield DatasetSubsets(self.generator(name), name)