import os
import math
import random
import numpy as np
import pandas as pd
from typing import List
from fastapi import Body
from typing import Union
from random import sample
from fastapi import FastAPI
from datetime import datetime
from dotenv import load_dotenv
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

def sampProduct(state_id, nproducts=N_PRODUCT, epsilon=EPSILON):
    sorted_policies = sorted(polDic[state_id].items(), key=lambda kv: kv[1], reverse=True)
    topProducts = [prod[0] for prod in sorted_policies[:nproducts]]
    seg_products = []

    while len(seg_products) < nproducts:
        probability = np.random.rand()
        if probability >= epsilon and topProducts:
            seg_products.append(topProducts.pop(0))
        else:
            available_products = list(rewardFull['product_title'].unique())
            available_products = [p for p in available_products if p not in seg_products]
            if available_products:
                seg_products.append(sample(available_products, 1)[0])
        seg_products = list(OrderedDict.fromkeys(seg_products))
    return seg_products

def nudgeMessage(product, content):
    return f"{content}"

@app.get("/two-tower-data")
def TwoTowerData():
    response = supabase.table(TWO_TOWER_TABLE).select("*", count="exact").execute()
    return response

@app.get("/interaction-log-data")
def InteractionLogData():
    response = supabase.table(INTERACTION_LOG_TABLE).select("*", count="exact").execute()
    return response

@app.get("/interaction-log-data/{state_id}")
def InteractionLogDataByStateId(state_id: str):
    response = supabase.table(INTERACTION_LOG_TABLE).select("*").eq("state_id", state_id).execute()
    return response.data

@app.get("/product-recommendation/{state_id}")
def ProductRecommendationByStateId(state_id:str):
    # Data awal
    raw_data = TwoTowerData().data
    data = pd.DataFrame(raw_data)
    data['OverallScore'] = data['score']
    rewardFull = data.groupby(['user_id', 'product_title', 'content'])['OverallScore'].sum().reset_index()

    # Data interaction
    raw_interaction_log = InteractionLogData().data
    interaction_log = pd.DataFrame(raw_interaction_log).to_dict('records')

    # Dictionaries utama
    countDic, polDic, rewDic, recoCount = {}, {}, {}, {}

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

        while len(seg_products) < nproducts:
            probability = np.random.rand()
            if probability >= epsilon and topProducts:
                seg_products.append(topProducts.pop(0))
            else:
                available_products = list(rewardFull['product_title'].unique())
                available_products = [p for p in available_products if p not in seg_products]
                if available_products:
                    seg_products.append(sample(available_products, 1)[0])
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
def BuyProductByStateId(state_id:str, buy_list: List[str] = Body(default=["Asuransi Kesehatan", "Produk Emas"])):
    # Data awal
    raw_data = TwoTowerData().data
    data = pd.DataFrame(raw_data)
    data['OverallScore'] = data['score']
    rewardFull = data.groupby(['user_id', 'product_title', 'content'])['OverallScore'].sum().reset_index()

    # Data interaction
    raw_interaction_log = InteractionLogData().data
    interaction_log = pd.DataFrame(raw_interaction_log).to_dict('records')

    # Dictionaries utama
    countDic, polDic, rewDic, recoCount = {}, {}, {}, {}

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

        while len(seg_products) < nproducts:
            probability = np.random.rand()
            if probability >= epsilon and topProducts:
                seg_products.append(topProducts.pop(0))
            else:
                available_products = list(rewardFull['product_title'].unique())
                available_products = [p for p in available_products if p not in seg_products]
                if available_products:
                    seg_products.append(sample(available_products, 1)[0])
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
        # Simpan reward/policy sebelum update
        reward_before = [rewDic[state_id].get(p, 0) for p in custList]
        policy_before = [polDic[state_id].get(p, 0) for p in custList]

        # Update nilai
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

            epsilon = max(0.01, epsilon * 0.95)

        # Simpan reward/policy setelah update
        reward_after = [rewDic[state_id][p] for p in custList]
        policy_after = [polDic[state_id][p] for p in custList]

        # Buat next recommendation satu kali
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
            "next_recommended": next_recommended
        }

        # Kirim ke Supabase
        supabase.table("interaction_log").insert(interaction_entry).execute()

        return next_recommended, epsilon
    
    seg_products, epsilon = valueUpdater(seg_products, 5, buy_list, epsilon)
    
    return {"update_rewards":dict(sorted(rewDic[state_id].items(), key=lambda x: x[1], reverse=True)),
    "update_policies": dict(sorted(polDic[state_id].items(), key=lambda x: x[1], reverse=True))}