from supabase import create_client, Client
from fastapi import FastAPI
from typing import Union
import random
import math
import os

app = FastAPI()

url: str = "https://tzopfeekvuztbabqbtmh.supabase.co"
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR6b3BmZWVrdnV6dGJhYnFidG1oIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcyOTAyMjI3MiwiZXhwIjoyMDQ0NTk4MjcyfQ.gDfOTfCmogVIFwerLjCMY1b5YI5DvfE3OQlrrA9seCM"
supabase: Client = create_client(url, key)

arms = {}         # Each key is a product_id with aggregated stats and bandit variables
total_pulls = 0   # Total number of recommendations made

def init_arms():
    """
    Load historical product data from Supabase's "two_tower_training_result" table,
    aggregate scores by product_id, and initialize each arm with a baseline "similarity"
    along with counters for the multi-armed bandit.
    """
    global arms
    response = supabase.table("two_tower_training_result").select("*").execute()
    data = response.data

    # Aggregate data by product_id
    for record in data:
        pid = record["product_id"]
        score = record["score"]
        if pid not in arms:
            arms[pid] = {"score_sum": score, "count": 1}
        else:
            arms[pid]["score_sum"] += score
            arms[pid]["count"] += 1

    # Finalize arms by computing average score (baseline similarity)
    # and initializing bandit variables
    for pid, stats in arms.items():
        stats["similarity"] = stats["score_sum"] / stats["count"]
        stats["n"] = 0             # Number of times this product has been recommended
        stats["total_reward"] = 0.0  # Cumulative reward
        del stats["score_sum"]
        del stats["count"]

# Initialize arms on startup
init_arms()

def select_arm_ucb() -> str:
    """
    Select a product (arm) using the UCB1 strategy.
    Returns an arm that has not been tried yet, or the arm with the highest UCB value.
    """
    global total_pulls, arms

    # Explore: Return any arm that hasn't been recommended yet
    for arm, stats in arms.items():
        if stats["n"] == 0:
            return arm

    # Exploit: Compute UCB for each arm and return the one with the highest value
    best_ucb = -float("inf")
    best_arm = None
    for arm, stats in arms.items():
        avg_reward = stats["total_reward"] / stats["n"]
        ucb = avg_reward + math.sqrt(2 * math.log(total_pulls + 1) / stats["n"])
        if ucb > best_ucb:
            best_ucb = ucb
            best_arm = arm
    return best_arm

# -------------------------
# FastAPI Endpoints
# -------------------------
@app.get("/")
def read_root():
    """
    Root endpoint: returns all rows from the "two_tower_training_result" table.
    """
    response = supabase.table("two_tower_training_result").select("*").execute()
    return response.data

@app.get("/recommend")
def get_recommendation():
    """
    Recommendation endpoint:
    Uses UCB1 to select a product recommendation for a new user.
    The recommendation is based on the historical aggregated data.
    It also simulates a reward (using Gaussian noise) to update the bandit state.
    """
    global total_pulls, arms
    total_pulls += 1

    # Select a product using the UCB1 algorithm
    chosen_product = select_arm_ucb()

    # Simulate a reward using the product's baseline similarity with some noise
    base_reward = arms[chosen_product]["similarity"]
    reward = random.gauss(mu=base_reward, sigma=0.1)

    # Update bandit counters for the chosen product
    arms[chosen_product]["n"] += 1
    arms[chosen_product]["total_reward"] += reward

    return {
        "recommended_product": chosen_product,
        "simulated_reward": reward,
        "arm_stats": arms[chosen_product],
        "total_pulls": total_pulls
    }
