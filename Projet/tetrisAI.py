import numpy
import tetris

class tetrisAI(object):
    def __init__(self):
        self.thetas = numpy.ones((2 * tetris.BOARD_WIDTH + 2, 1)) 
        self.zs = numpy.zeros((2 * tetris.BOARD_WIDTH + 2, 1)) 
        self.deltas = numpy.zeros((2 * tetris.BOARD_WIDTH + 2, 1))
        self.beta = 1
        self.t = 0

    def apply_policy(self):
        features = tetris.board.get_features()

    def update_deltas(self, features, chosen_features_index):
        chosen_features = features[chosen_features_index]
        features_esperance = numpy.sum(features, axis=1)
        score_ratio =  chosen_features - features_esperance

        self.zs = self.beta * self.zs + score_ratio
        self.deltas = self.deltas + t / (t + 1) * (chosen_features[REWARD_INDEX] * self.zs - self.deltas)

    def update_thetas(self):
        self.thetas = self.thetas + self.alpha * self.deltas
        self.reset_deltas()

    def reset_deltas(self):
        self.deltas = self.deltas / 2

    def choose_action(self):
        features_list, actions_list = tetris.board.get_features()
        features = numpy.array(features_list)
        
        quality_function = self.deltas.T * features