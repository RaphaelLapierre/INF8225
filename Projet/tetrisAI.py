import numpy
import tetris


class TetrisAI:
    def __init__(self):
        self.thetas = numpy.ones((2 * tetris.BOARD_WIDTH + 2, 1)) 
        self.zs = numpy.zeros((2 * tetris.BOARD_WIDTH + 2, 1)) 
        self.deltas = numpy.zeros((2 * tetris.BOARD_WIDTH + 2, 1))
        self.beta = 0.5
        self.alpha = 0.000000001
        self.t = 0

    def apply_policy(self):
        features = tetris.board.get_features()

    def update_deltas(self, features, chosen_features_index):
        chosen_features = features[:,chosen_features_index]
        features_esperance = numpy.sum(features, axis=1)
        score_ratio =  chosen_features - features_esperance

        self.zs = self.beta * self.zs + score_ratio
        self.deltas = self.deltas + float(self.t) / (self.t + 1) * (chosen_features[tetris.REWARD_INDEX] * self.zs - self.deltas)
        self.t += 1

    def update_thetas(self):
        self.thetas = self.thetas + self.alpha * self.deltas
        self.reset_deltas()

    def reset_deltas(self):
        self.deltas = self.deltas / 2

    def choose_action(self):
        features_actions = tetris.board.get_features()
        features = []
        actions = []
        for i in range(len(features_actions)):
            features.append(features_actions[i][0])
            actions.append(features_actions[i][1])
        matrix_features = numpy.array(features)
        Q = numpy.dot(self.thetas.T, matrix_features.T)
        action_index = numpy.argmax(Q)
        if(action_index >= len(actions)):
            return
        x, y, rotation = actions[action_index]
        deltaX = x - tetris.board.active_shape.x
        for i in range(rotation):
            tetris.board.active_shape.rotate()
        if(deltaX > 0):
            for i in range(deltaX):
                tetris.board.move_right()
        else:
            for i in range(abs(deltaX)):
                tetris.board.move_left()
        for i in range(y):
            tetris.board.move_down()

        #tetris.board.active_shape.x = x
        #tetris.board.active_shape.y = y

        self.update_deltas(matrix_features.T, action_index)
