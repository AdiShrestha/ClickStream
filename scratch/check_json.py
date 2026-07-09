import json, numpy as np

with open("results/week4_extension/sweep_results.json", "r") as f:
    data = json.load(f)

for k, v in data.items():
    print(f"{k}: attack_mean={v['mean_attack_attacker_delta']*100:.2f}pp, benign_mean={v['mean_benign_attacker_delta']*100:.2f}pp, gap={v['gap']*100:.2f}pp")
