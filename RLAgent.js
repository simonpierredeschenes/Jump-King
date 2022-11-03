/*
 * The agent is called 5 times per second approximately
 * Each time, it must compute the next action based on its historic
 *
 * A state is a vector of size 5
 *   The first element is the global height of the player in the game (0 is at the bottom)
 *   The second element is the x position of the player in the screen (0 is at the left)
 *   The third element is the y position of the player in the screen (0 is at the top)
 *   The fourth element is true if the player is touching the ground, false otherwise
 *   The fifth element is a vector containing all the solid edges in the screen
 * 
 * An action is a vector of size 2
 *   The first element is either -1 to move left, 0 for no lateral movement or 1 to move right
 *   The second element is true to jump, false otherwise
 * 
 * A historic entry is a vector of size 4
 *   The first element is the previous state
 *   The second element is the previous action
 *   The third element is the reward received after the previous action
 *   The fourth element is the next state
 */
class RLAgent {
    constructor() {
        this.historic = []
    }

    addEntryToHistoric(previousState, previousAction, reward, nextState) {
        this.historic.push([previousState, previousAction, reward, nextState]);
    }

    /* This function is called 5 times per second approximately
     * Its goal is to compute the next action based on the agent's historic
     * The returned action is a vector of size 2
     *   The first element is either -1 to move left, 0 for no lateral movement or 1 to move right
     *   The second element is true to jump, false otherwise
     */
    chooseAction() {
	let direction = -1;
	if (this.historic.length >= 5) {
            direction = this.historic[this.historic.length-1][1][0] * -1;
            for (let i = 1; i < 5; i++) {
                if (this.historic[this.historic.length-1][1][0] != this.historic[this.historic.length-1-i][1][0]) {
                    direction = this.historic[this.historic.length-1][1][0];
                    break;
                }
	    }
	}

        let jump = true;
        if (this.historic.length >= 4) {
            jump = !this.historic[this.historic.length-1][1][1];
            for (let i = 1; i < 4; i++) {
                if (this.historic[this.historic.length-1][1][1] != this.historic[this.historic.length-1-i][1][1]) {
                    jump = this.historic[this.historic.length-1][1][1];
                    break;
                }
            }
        }

        return [direction, jump];
    }
}
