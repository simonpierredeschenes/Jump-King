class Demonstration:
    def __init__(self):
        self.actions = [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 0], \
                        [0, 0], [0, 0], [0, 0], [0, 0], \
 \
                        [-1, 0], [-1, 0], [-1, 0], [-1, 0], [-1, 0], [-1, 0], \
                        [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], \
 \
                        [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 0], \
                        [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], \
 \
                        [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 0], \
                        [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], \
 \
                        [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 0], \
                        [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], \
 \
                        [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], \
                        [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], \
 \
                        [-1, 1], [-1, 1], [-1, 1], [-1, 0], \
                        [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], \
 \
                        [1, 0], [1, 0], [1, 0], \
                        [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], \
 \
                        [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 0], \
                        [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], \
 \
                        [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 0], \
                        [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], \
 \
                        [1, 0], [1, 0], [1, 0], [1, 0], \
                        [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], \
 \
                        [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 0], \
                        [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], \
 \
                        [1, 0], [1, 0], [1, 0], \
                        [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], \
 \
                        [1, 1], [1, 0], \
                        [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], \
 \
                        [1, 0], [1, 0], [1, 0], \
                        [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], \
 \
                        [1, 1], [1, 1], [1, 1], [1, 1], [1, 0], \
                        [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], \
 \
                        [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 0], \
                        [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], \
 \
                        [1, 0], [1, 0], [1, 0], \
                        [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], \
 \
                        [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 0], \
                        [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], \
 \
                        [-1, 0], [-1, 0], \
                        [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], \
 \
                        [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 0], [0, 0], \
                        [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], \
 \
                        [-1, 0], [-1, 0], \
                        [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], \
 \
                        [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 0], \
                        [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
        self.action_index = 0

    def get_next_action(self):
        action = None
        if self.action_index < len(self.actions):
            action = self.actions[self.action_index]
            self.action_index += 1
        return action
