import asyncio
import websockets
from agent import Agent

agent = Agent()
steps_since_reset = 0
episode = 0
global epsilon
epsilon=0.95

def parse_lines(lines_str):
    lines = []
    current_index = 1
    while current_index < len(lines_str):
        line_start_index = current_index
        line_end_index = lines_str.find("]", line_start_index) + 1
        line_str = lines_str[line_start_index:line_end_index]

        x1_start_index = 1
        x1_end_index = line_str.find(",", x1_start_index)
        x1_str = line_str[x1_start_index:x1_end_index]
        x1 = float(x1_str)

        y1_start_index = x1_end_index + 1
        y1_end_index = line_str.find(",", y1_start_index)
        y1_str = line_str[y1_start_index:y1_end_index]
        y1 = float(y1_str)

        x2_start_index = y1_end_index + 1
        x2_end_index = line_str.find(",", x2_start_index)
        x2_str = line_str[x2_start_index:x2_end_index]
        x2 = float(x2_str)

        y2_start_index = x2_end_index + 1
        y2_end_index = line_str.find("]", y2_start_index)
        y2_str = line_str[y2_start_index:y2_end_index]
        y2 = float(y2_str)

        lines.append((x1, y1, x2, y2))
        current_index = line_end_index + 1
    return lines


def parse_state(state_str):
    global_height_start_index = 1
    global_height_end_index = state_str.find(",", global_height_start_index)
    global_height_str = state_str[global_height_start_index:global_height_end_index]
    global_height = float(global_height_str)

    player_x_start_index = global_height_end_index + 1
    player_x_end_index = state_str.find(",", player_x_start_index)
    player_x_str = state_str[player_x_start_index:player_x_end_index]
    player_x = float(player_x_str)

    player_y_start_index = player_x_end_index + 1
    player_y_end_index = state_str.find(",", player_y_start_index)
    player_y_str = state_str[player_y_start_index:player_y_end_index]
    player_y = float(player_y_str)

    is_on_ground_start_index = player_y_end_index + 1
    is_on_ground_end_index = state_str.find(",", is_on_ground_start_index)
    is_on_ground_str = state_str[is_on_ground_start_index:is_on_ground_end_index]
    is_on_ground = is_on_ground_str == "true"

    jump_counter_start_index = is_on_ground_end_index + 1
    jump_counter_end_index = state_str.find(",", jump_counter_start_index)
    jump_counter_str = state_str[jump_counter_start_index:jump_counter_end_index]
    jump_counter = int(jump_counter_str)

    lines_start_index = jump_counter_end_index + 1
    current_index = lines_start_index + 1
    nb_open_brackets = 1
    while current_index < len(state_str) and nb_open_brackets > 0:
        if state_str[current_index] == "[":
            nb_open_brackets += 1
        elif state_str[current_index] == "]":
            nb_open_brackets -= 1
        current_index += 1
    lines_end_index = current_index
    lines_str = state_str[lines_start_index:lines_end_index]
    lines = parse_lines(lines_str)

    return global_height, player_x, player_y, is_on_ground, jump_counter, lines


def parse_action(action_str):
    direction_start_index = 1
    direction_end_index = action_str.find(",", direction_start_index)
    direction_str = action_str[direction_start_index:direction_end_index]
    direction = int(direction_str)

    jump_start_index = direction_end_index + 1
    jump_end_index = action_str.find("]", jump_start_index)
    jump_str = action_str[jump_start_index:jump_end_index]
    jump = jump_str == "true"

    return direction, jump


def parse_historic_entry(historic_entry_str):
    previous_state_start_index = 1
    current_index = previous_state_start_index + 1
    nb_open_brackets = 1
    while current_index < len(historic_entry_str) and nb_open_brackets > 0:
        if historic_entry_str[current_index] == "[":
            nb_open_brackets += 1
        elif historic_entry_str[current_index] == "]":
            nb_open_brackets -= 1
        current_index += 1
    previous_state_end_index = current_index
    previous_state_str = historic_entry_str[previous_state_start_index:previous_state_end_index]
    previous_state = parse_state(previous_state_str)

    action_start_index = previous_state_end_index + 1
    current_index = action_start_index + 1
    nb_open_brackets = 1
    while current_index < len(historic_entry_str) and nb_open_brackets > 0:
        if historic_entry_str[current_index] == "[":
            nb_open_brackets += 1
        elif historic_entry_str[current_index] == "]":
            nb_open_brackets -= 1
        current_index += 1
    action_end_index = current_index
    action_str = historic_entry_str[action_start_index:action_end_index]
    action = parse_action(action_str)

    reward_start_index = action_end_index + 1
    reward_end_index = historic_entry_str.find(",", reward_start_index)
    reward_str = historic_entry_str[reward_start_index:reward_end_index]
    reward = float(reward_str)

    next_state_start_index = reward_end_index + 1
    current_index = next_state_start_index + 1
    nb_open_brackets = 1
    while current_index < len(historic_entry_str) and nb_open_brackets > 0:
        if historic_entry_str[current_index] == "[":
            nb_open_brackets += 1
        elif historic_entry_str[current_index] == "]":
            nb_open_brackets -= 1
        current_index += 1
    next_state_end_index = current_index
    next_state_str = historic_entry_str[next_state_start_index:next_state_end_index]
    next_state = parse_state(next_state_str)

    return previous_state, action, reward, next_state


def action_to_string(action):
    action_str = "ACT["
    action_str += str(action[0]) + ","
    action_str += ("true" if action[1] else "false") + "]"
    return action_str


async def on_receive(websocket):
    async for historic_entry_str in websocket:
        historic_entry = parse_historic_entry(historic_entry_str)
        agent.add_entry_to_historic(*historic_entry)

        global steps_since_reset
        if steps_since_reset < 300:
            steps_since_reset += 1
            action = agent.choose_action()
            response = action_to_string(action)
        else:
            steps_since_reset = 0
            response = "RST"
            global episode
            episode += 1

            if episode == 1:
                with open("dql_episode.csv", "w+", newline="") as file:
                    file.write("Episode,cumulative_reward,loss\n")
                    file.write(str(episode) + "," + str(agent.G) + "," + str(agent.last_loss_episode) + "\n")
                    print(episode, agent.G, agent.last_loss_episode)
            else:
                with open("dql_episode.csv", "a", newline="") as file:
                    file.write(str(episode) + "," + str(agent.G) + "," + str(agent.last_loss_episode) + "\n")
                    print(episode, agent.G, agent.last_loss_episode)
            agent.G = 0
            # #On reset le epsion
            # agent.epsilon=epsilon**(episode)
        await websocket.send(response)


async def main():
    async with websockets.serve(on_receive, "localhost", 65432):
        await asyncio.Future()


asyncio.run(main())
