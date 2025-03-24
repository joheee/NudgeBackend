import os
import math
import random
import numpy as np
import pandas as pd
from typing import List
from fastapi import Body
from random import sample
from fastapi import FastAPI
from datetime import datetime
from dotenv import load_dotenv
from typing import Union, Optional
from collections import OrderedDict
from supabase import create_client, Client

app = FastAPI()
load_dotenv()

# Supabase Initialization
url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Constants for table names
TWO_TOWER_TABLE = "temp_ik_resultsnudgetwscorefinal"
INTERACTION_LOG_TABLE = "interaction_log"
N_PRODUCT = 5
EPSILON = 0.01

# Utility function
def GaussianDistribution(loc, scale, size):
    return np.random.normal(loc, scale, size)

def interaction_log_data():
    response = supabase.table(INTERACTION_LOG_TABLE).select("*", count="exact").execute()
    return response

def interaction_log_data_bystate_id(state_id: str):
    response = supabase.table(INTERACTION_LOG_TABLE).select("*").eq("state_id", state_id).execute()
    return response.data

def two_tower_data(risk_level: Optional[str] = "High"):
    if risk_level == "High":
        risk_filter = ["High", "Moderate", "Conventional"]
    elif risk_level == "Moderate":
        risk_filter = ["Moderate", "Conventional"]
    elif risk_level == "Conventional":
        risk_filter = ["Conventional"]
    else:
        risk_filter = None

    query = supabase.table(TWO_TOWER_TABLE).select("*", count="exact")
    if risk_filter:
        query = query.in_("risklevel", risk_filter)
    
    response = query.execute()
    return response

@app.get("/two-tower-data-count")
def two_tower_data_count(risk_level: Optional[str] = "High"):
    if risk_level == "High":
        risk_filter = ["High", "Moderate", "Conventional"]
    elif risk_level == "Moderate":
        risk_filter = ["Moderate", "Conventional"]
    elif risk_level == "Conventional":
        risk_filter = ["Conventional"]
    else:
        risk_filter = None

    query = supabase.table(TWO_TOWER_TABLE).select("*", count="exact")
    if risk_filter:
        query = query.in_("risklevel", risk_filter)
    
    response = query.execute()
    return {
        'risk_level': risk_level,
        'risk_filter': risk_filter,
        'count': response.count
    }

@app.get("/product-recommendation/{state_id}")
def product_recommendation_by_state_id(state_id:str, risk_level: Optional[str] = "High"):
    # Data awal
    raw_data = two_tower_data(risk_level).data
    data = pd.DataFrame(raw_data)
    data['OverallScore'] = data['score']
    rewardFull = data.groupby(['user_id', 'product_title', 'content'])['OverallScore'].sum().reset_index()

    # Data interaction
    raw_interaction_log = interaction_log_data().data
    interaction_log = pd.DataFrame(raw_interaction_log).to_dict('records')
    
    countDic, polDic, rewDic, recoCount = {}, {}, {}, {}
    cumulative_reward_log = []
    cumulative_regret_log = []
    optimal_arm_counts = []

    avg_rewards_per_product = rewardFull.groupby('product_title')['OverallScore'].mean().to_dict()
    best_expected_reward = max(avg_rewards_per_product.values())
    best_product = max(avg_rewards_per_product, key=avg_rewards_per_product.get)

    # Setup awal dari data
    users = list(rewardFull.user_id.unique())
    for id in users:
        subset = rewardFull[rewardFull['user_id'] == id]
        countDic[id] = {row['product_title']: int(row['OverallScore']) for _, row in subset.iterrows()}

    if state_id not in countDic:
        countDic[state_id] = {}
    if state_id not in polDic:
        polDic[state_id] = {}
    if state_id not in rewDic:
        rewDic[state_id] = {}
    if state_id not in recoCount:
        recoCount[state_id] = {}

    prodCounts = countDic[state_id]
    for pkey in prodCounts.keys():
        if pkey not in polDic[state_id]:
            polDic[state_id][pkey] = GaussianDistribution(loc=prodCounts[pkey], scale=1, size=1)[0].round(2)
        if pkey not in rewDic[state_id]:
            rewDic[state_id][pkey] = GaussianDistribution(loc=prodCounts[pkey], scale=1, size=1)[0].round(2)

    def sampProduct(nproducts, state_id, epsilon):
        sorted_policies = sorted(polDic[state_id].items(), key=lambda kv: kv[1], reverse=True)
        topProducts = [prod[0] for prod in sorted_policies[:nproducts]]
        seg_products = []
    
        # Tambahkan best arm di awal jika belum ada
        if best_product in polDic[state_id] and best_product not in seg_products:
            seg_products.append(best_product)
    
        while len(seg_products) < nproducts:
            probability = np.random.rand()
            if probability >= epsilon and topProducts:
                next_prod = topProducts.pop(0)
            else:
                available_products = list(rewardFull['product_title'].unique())
                available_products = [p for p in available_products if p not in seg_products]
                if available_products:
                    next_prod = sample(available_products, 1)[0]
                else:
                    break
    
            if next_prod not in seg_products:
                seg_products.append(next_prod)
    
        seg_products = list(OrderedDict.fromkeys(seg_products))
        return seg_products

    nProducts = 5
    epsilon = 0.01

    previous_entries = [entry for entry in interaction_log if entry['state_id'] == state_id]
    if previous_entries:
        last_entry = previous_entries[-1]
        next_data = last_entry['next_recommended']
        try:
            if isinstance(next_data, str):
                seg_products = eval(next_data)
            elif isinstance(next_data, list):
                seg_products = next_data
            else:
                seg_products = []
        except Exception as e:
            print(f"Gagal memuat rekomendasi sebelumnya: {e}")
            seg_products = []
        print(f"(Menggunakan rekomendasi lanjutan dari interaksi sebelumnya untuk state {state_id})")
    else:
        seg_products = sampProduct(nProducts, state_id, epsilon)

    return seg_products

@app.post("/product-recommendation/{state_id}")
def buy_product_by_state_id(state_id:str, buy_list: List[str] = Body(default=["Tabungan Haji"])):
    # Data awal
    raw_data = two_tower_data().data
    data = pd.DataFrame(raw_data)
    data['OverallScore'] = data['score']
    rewardFull = data.groupby(['user_id', 'product_title', 'content'])['OverallScore'].sum().reset_index()

    # Data interaction
    raw_interaction_log = interaction_log_data().data
    interaction_log = pd.DataFrame(raw_interaction_log).to_dict('records')
    
    countDic, polDic, rewDic, recoCount = {}, {}, {}, {}
    cumulative_reward_log = []
    cumulative_regret_log = []
    optimal_arm_counts = []

    avg_rewards_per_product = rewardFull.groupby('product_title')['OverallScore'].mean().to_dict()
    best_expected_reward = max(avg_rewards_per_product.values())
    best_product = max(avg_rewards_per_product, key=avg_rewards_per_product.get)

    # Setup awal dari data
    users = list(rewardFull.user_id.unique())
    for id in users:
        subset = rewardFull[rewardFull['user_id'] == id]
        countDic[id] = {row['product_title']: int(row['OverallScore']) for _, row in subset.iterrows()}

    if state_id not in countDic:
        countDic[state_id] = {}
    if state_id not in polDic:
        polDic[state_id] = {}
    if state_id not in rewDic:
        rewDic[state_id] = {}
    if state_id not in recoCount:
        recoCount[state_id] = {}

    prodCounts = countDic[state_id]
    for pkey in prodCounts.keys():
        if pkey not in polDic[state_id]:
            polDic[state_id][pkey] = GaussianDistribution(loc=prodCounts[pkey], scale=1, size=1)[0].round(2)
        if pkey not in rewDic[state_id]:
            rewDic[state_id][pkey] = GaussianDistribution(loc=prodCounts[pkey], scale=1, size=1)[0].round(2)

    def sampProduct(nproducts, state_id, epsilon):
        sorted_policies = sorted(polDic[state_id].items(), key=lambda kv: kv[1], reverse=True)
        topProducts = [prod[0] for prod in sorted_policies[:nproducts]]
        seg_products = []
    
        # Tambahkan best arm di awal jika belum ada
        if best_product in polDic[state_id] and best_product not in seg_products:
            seg_products.append(best_product)
    
        while len(seg_products) < nproducts:
            probability = np.random.rand()
            if probability >= epsilon and topProducts:
                next_prod = topProducts.pop(0)
            else:
                available_products = list(rewardFull['product_title'].unique())
                available_products = [p for p in available_products if p not in seg_products]
                if available_products:
                    next_prod = sample(available_products, 1)[0]
                else:
                    break
    
            if next_prod not in seg_products:
                seg_products.append(next_prod)
    
        seg_products = list(OrderedDict.fromkeys(seg_products))
        return seg_products

    nProducts = 5
    epsilon = 0.01

    previous_entries = [entry for entry in interaction_log if entry['state_id'] == state_id]
    if previous_entries:
        last_entry = previous_entries[-1]
        next_data = last_entry['next_recommended']
        try:
            if isinstance(next_data, str):
                seg_products = eval(next_data)
            elif isinstance(next_data, list):
                seg_products = next_data
            else:
                seg_products = []
        except Exception as e:
            print(f"Gagal memuat rekomendasi sebelumnya: {e}")
            seg_products = []
        print(f"(Menggunakan rekomendasi lanjutan dari interaksi sebelumnya untuk state {state_id})")
    else:
        seg_products = sampProduct(nProducts, state_id, epsilon)

    def valueUpdater(seg_products, loc, custList, epsilon):
        reward_before = [rewDic[state_id].get(p, 0) for p in custList]
        policy_before = [polDic[state_id].get(p, 0) for p in custList]

        total_reward_this_round = 0.0
        regret_this_round = 0.0
        picked_best = False

        for prod in custList:
            if prod not in polDic[state_id]:
                polDic[state_id][prod] = rewDic[state_id].get(prod, 0)

            rewDic[state_id].setdefault(prod, 0)
            polDic[state_id].setdefault(prod, 0)
            recoCount[state_id].setdefault(prod, 1)

            rew = GaussianDistribution(loc=loc, scale=0.5, size=1)[0].round(2)
            rewDic[state_id][prod] += rew
            polDic[state_id][prod] += (1 / recoCount[state_id][prod]) * (rew - polDic[state_id][prod])
            recoCount[state_id][prod] += 1

            total_reward_this_round += rew
            expected_reward = avg_rewards_per_product.get(prod, 0)
            regret_this_round += (best_expected_reward - expected_reward)

            if prod == best_product:
                picked_best = True

            epsilon = max(0.01, epsilon * 0.95)

        reward_after = [rewDic[state_id][p] for p in custList]
        policy_after = [polDic[state_id][p] for p in custList]
        next_recommended = sampProduct(nProducts, state_id, epsilon)

        interaction_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "state_id": state_id,
            "product_offered": seg_products,
            "product_bought": custList,
            "reward_before": reward_before,
            "policy_before": policy_before,
            "reward_after": reward_after,
            "policy_after": policy_after,
            "next_recommended": next_recommended,
            "total_reward": total_reward_this_round,
            "cumulative_regret": regret_this_round,
            "picked_best_arm": picked_best
        }

        interaction_log.append(interaction_entry)
        cumulative_reward_log.append(total_reward_this_round)
        cumulative_regret_log.append(regret_this_round)
        optimal_arm_counts.append(int(picked_best))

        supabase.table("interaction_log").insert({
            "timestamp": interaction_entry["timestamp"],
            "state_id": interaction_entry["state_id"],
            "product_offered": interaction_entry["product_offered"],
            "product_bought": interaction_entry["product_bought"],
            "reward_before": interaction_entry["reward_before"],
            "policy_before": interaction_entry["policy_before"],
            "reward_after": interaction_entry["reward_after"],
            "policy_after": interaction_entry["policy_after"],
            "next_recommended": interaction_entry["next_recommended"]
        }).execute()

        return next_recommended, epsilon
    
    seg_products, epsilon = valueUpdater(seg_products, 5, buy_list, epsilon)
    n_new = len(cumulative_reward_log)
    new_interactions = interaction_log[-n_new:] 

    reward_policy_df = pd.DataFrame({
        "product_title": list(rewDic[state_id].keys()),
        "updated_reward": list(rewDic[state_id].values()),
        "updated_policy": [polDic[state_id].get(p, 0) for p in rewDic[state_id].keys()],
        "state_id": state_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })    

    performance_df = pd.DataFrame({
        "timestamp": [entry["timestamp"] for entry in new_interactions],
        "state_id": [entry["state_id"] for entry in new_interactions],
        "total_reward": cumulative_reward_log,
        "cumulative_regret": cumulative_regret_log,
        "picked_best_arm": optimal_arm_counts
    })

    reward_policy_data = reward_policy_df.to_dict(orient='records')
    supabase.table("reward_policy_log").insert(reward_policy_data).execute()

    performance_data = performance_df.to_dict(orient="records")
    supabase.table("performance_log").insert(performance_data).execute()

    return {
        "update_rewards":dict(sorted(rewDic[state_id].items(), key=lambda x: x[1], reverse=True)),
        "update_policies": dict(sorted(polDic[state_id].items(), key=lambda x: x[1], reverse=True))
    }