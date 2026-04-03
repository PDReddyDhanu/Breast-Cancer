import pandas as pd
import random

NUM_ROWS = 3000   # How many patients

data = []

for i in range(1, NUM_ROWS + 1):

    pid = f"P{i:05d}"

    # Basic info
    age = random.randint(25, 75)
    stage = random.randint(1, 4)

    # QoL scores
    fatigue = random.randint(20, 95)
    pain = random.randint(20, 90)
    emotion = random.randint(25, 90)
    physical = random.randint(30, 90)
    social = random.randint(30, 90)
    cognitive = random.randint(30, 90)
    sleep = random.randint(20, 90)
    appetite = random.randint(20, 85)

    prev_nausea = random.randint(0, 1)
    prev_neuro = random.randint(0, 1)

    # Rule for output
    if fatigue > 75 and pain > 70:
        side = "Fatigue"
        sev = "High"
        risk = "High"

    elif prev_nausea == 1:
        side = "Nausea"
        sev = "Medium"
        risk = "Medium"

    elif prev_neuro == 1:
        side = "Neuropathy"
        sev = "Medium"
        risk = "Medium"

    else:
        side = "None"
        sev = "Low"
        risk = "Low"


    data.append([
        pid, age, stage, fatigue, pain, emotion,
        physical, social, cognitive, sleep, appetite,
        prev_nausea, prev_neuro,
        side, sev, risk
    ])


columns = [
    "id","age","stage","fatigue","pain","emotion",
    "physical","social","cognitive","sleep","appetite",
    "prev_nausea","prev_neuropathy",
    "side_effect","severity","risk"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("raw_qol_data.csv", index=False)

print("Dataset created:", NUM_ROWS)
