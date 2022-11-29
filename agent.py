# The agent is called 5 times per second approximately
# Each time, it must compute the next action based on its historic
#
# A state is a vector of size 5
#   The first element is the global height of the player in the game (0 is at the bottom)
#   The second element is the x position of the player in the screen (0 is at the left)
#   The third element is the y position of the player in the screen (0 is at the top)
#   The fourth element is true if the player is touching the ground, false otherwise
#   The fifth element is a vector containing all the solid edges in the screen under the form (x1, y1, x2, y2)
#
# An action is a vector of size 2
#   The first element is either -1 to move left, 0 for no lateral movement or 1 to move right
#   The second element is true to jump, false otherwise
#
# A historic entry is a vector of size 4
#   The first element is the previous state
#   The second element is the previous action
#   The third element is the reward received after the previous action
#   The fourth element is the next state


class Agent:
    def __init__(self):
        self.historic = []

    def add_entry_to_historic(self, previous_state, action, reward, next_state):
        self.historic.append((previous_state, action, reward, next_state))

    def choose_action(self):
        direction = -1
        if len(self.historic) >= 5 and self.historic[-1][1][0] != 0:
            direction = self.historic[-1][1][0] * -1
            for i in range(5):
                if self.historic[-1][1][0] != self.historic[len(self.historic)-1-i][1][0]:
                    direction = self.historic[-1][1][0]
                    break

        jump = True
        if len(self.historic) >= 4:
            jump = not self.historic[-1][1][1]
            for i in range(4):
                if self.historic[-1][1][1] != self.historic[len(self.historic)-1-i][1][1]:
                    jump = self.historic[-1][1][1]
                    break

        return direction, jump
