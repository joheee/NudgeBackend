from fastapi import FastAPI
from typing import Union
import math
import random

app = FastAPI()

arms = {
    "product_A": {"n": 0, "total_reward": 0.0, "similarity": 0.8},
    "product_B": {"n": 0, "total_reward": 0.0, "similarity": 0.7},
    "product_C": {"n": 0, "total_reward": 0.0, "similarity": 0.9}
}

# Global counter for total recommendations (pulls)
total_pulls = 0

def select_arm_ucb() -> str:

    global total_pulls, arms
    # Check for arms that haven't been pulled yet.
    for arm, stats in arms.items():
        if stats["n"] == 0:
            return arm

    # Compute UCB for each arm and choose the best
    best_ucb = -float("inf")
    best_arm = None
    for arm, stats in arms.items():
        avg_reward = stats["total_reward"] / stats["n"]
        # Using total_pulls+1 to avoid log(0) issues (although total_pulls should be > 0 here)
        ucb = avg_reward + math.sqrt(2 * math.log(total_pulls + 1) / stats["n"])
        if ucb > best_ucb:
            best_ucb = ucb
            best_arm = arm
    return best_arm

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/recommend")
def get_recommendation():

    global total_pulls, arms
    # Increment the total number of pulls
    total_pulls += 1

    # Select an arm (product) using UCB1
    chosen_arm = select_arm_ucb()

    # Simulate a reward:
    # Here we use the arm's similarity value as a base reward and add some noise.
    base_reward = arms[chosen_arm]["similarity"]
    reward = random.gauss(mu=base_reward, sigma=0.1)

    # Update the arm's statistics with the new reward.
    arms[chosen_arm]["n"] += 1
    arms[chosen_arm]["total_reward"] += reward

    return {
        "recommended_product": chosen_arm,
        "reward": reward,
        "arm_stats": arms[chosen_arm],
        "total_pulls": total_pulls
    }

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
