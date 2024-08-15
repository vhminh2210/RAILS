from utils import split_data

class Arg:
    def __init__(self, root= 'datasets', dataset= 'd1'):
        self.root = root
        self.dataset = dataset

args = Arg()
split_data(args)