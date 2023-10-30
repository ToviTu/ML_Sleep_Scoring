import pandas as pd
import numpy as np
import boto3
from pathlib import Path
import glob
import re
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from LFP_util import data_manage as dm
import os

ENCODING = {"None": 0, "0-1": 1, "0-2": 2, "1-0": 3, "1-2": 4, "2-0": 5, "2-1": 6}
DECODING = {val: key for key, val in ENCODING.items()}
encode = lambda x: ENCODING[x]
decode = lambda x: DECODING[x]

ANIMALS_DATASET = [
    "CAF37_day1",
    "CAF78_day1",
    "KDR14_day1",
    "KDR27_day1",
    "KDR36_day1",
    "CAF26",
    "CAF34",
    "CAF42",
    "CAF69",
]

# ANIMALS_DATASET = [
#     "CAF37_day1",
#     "CAF78_day1",
#     "KDR14_day1",
#     "KDR27_day1",
#     "KDR36_day1",
#     "CAF49_day4",
#     "CAF78_day2",
#     "CAF49_day1",
#     "CAF49_day2",
#     "CAF62_day1",
#     "CAF82_day1",
#     "CAF62_NEW",
#     "CAF61_day1",
# ]

ANIMALS = [re.search(r"CAF\d+|KDR\d+", animal).group() for animal in ANIMALS_DATASET]
PRED_DIR = "./data/CNN_Pred/"
FLICKER_DIR = "./data/Flicker_data/"


def get_s3_client():
    with open(f"{Path.home()}/.aws/credentials", "r") as f:
        for line in f:
            if "aws_access_key_id" in line:
                id = re.search(r"\W(\w+)", line).group(1)
            if "aws_secret_access_key" in line:
                key = re.search(r"\W(\w+)", line).group(1)
        return boto3.Session(aws_access_key_id=id, aws_secret_access_key=key).client(
            "s3", endpoint_url="https://s3-central.nrp-nautilus.io"
        )


def get_predictions(animal, client):
    try:
        all_objects = client.list_objects_v2(
            Bucket="hengenlab", Prefix=f"{animal}/Runs/wnr-v14-perregion-c24k-0-64"
        )["Contents"]
    except:
        print(f"Files not found for {animal}")
        return []
    flatten_objects = [o["Key"] for o in all_objects]
    return [object for object in flatten_objects if "predictions" in object]


def get_flickers(animal, client):
    all_objects = client.list_objects_v2(
        Bucket="hengenlab",
        Prefix=f"{animal}/flicker-calling/Results/flicker-calling-narrownone-{animal}-wnr-v14-perregion-c24k",
    )["Contents"]
    flatten_objects = [o["Key"] for o in all_objects]
    return [
        object
        for object in flatten_objects
        if "flicker" in object and "transitions" not in object
    ]


def compute_hourly_rate(df):
    result = {
        "Hour": [],
        "None": [],
        "0-1": [],
        "0-2": [],
        "1-0": [],
        "1-2": [],
        "2-0": [],
        "2-1": [],
    }

    i = 0
    while i + 15 * 3600 < df.shape[0]:
        df_clip = df.iloc[i : i + 15 * 3600]
        rates = df_clip.groupby(df_clip).count()
        for type in ["None", "0-1", "0-2", "1-0", "1-2", "2-0", "2-1"]:
            result[type].append(rates[type] if type in rates.index.array else 0)
        result["Hour"].append(i // (15 * 3600))
        i += 15 * 3600
    temp = pd.DataFrame(result)
    return temp


def compute_percent_by_bout(flicker, raw):
    result = {"Type": [], "Percent": []}

    raw = raw["label_wnr_012"]
    df = np.append(["<s>"], raw)
    state_changes = np.append(np.where(df[:-1] != df[1:])[0], [raw.shape[0]])

    for start_1, start_2 in zip(state_changes[:-1], state_changes[1:]):
        clip = flicker[start_1:start_2]
        for type in clip.unique():
            result["Percent"].append((clip == type).mean())
            result["Type"].append(type)

    return pd.DataFrame(result)


def compute_hour_rate_by_bout(flicker, raw):
    flicker = np.array(list(map(encode, flicker)))
    result = {"Type": [], "Rate": [], "Bout_duration": []}

    raw = raw["label_wnr_012"]
    df = np.append(["<s>"], raw)
    state_changes = np.append(np.where(df[:-1] != df[1:])[0], [raw.shape[0]])

    for start_1, start_2 in zip(state_changes[:-1], state_changes[1:]):
        clip = flicker[start_1:start_2]
        flicker_bout_start = np.where(np.diff(clip, prepend=[-1]) != 0)[0]
        counter = {key: 0 for key in DECODING.keys()}
        for idx in flicker_bout_start:
            counter[flicker[idx]] += 1

        for type, count in counter.items():
            result["Type"].append(type)
            result["Bout_duration"].append(clip.shape[0] * 1 / 15)
            result["Rate"].append(count)

    return pd.DataFrame(result)


def compute_hour_rate(state, raw):
    state = np.array(list(map(encode, state)))

    flicker_bout_start = np.where(np.diff(state, prepend=[-1]) != 0)[0]
    result = {
        "Type": [],
        "Count": [],
    }
    head = 0
    step = 3600 * 15
    while head + step < flicker_bout_start[-1]:
        counter = {key: 0 for key in DECODING.keys()}
        for idx in flicker_bout_start:
            counter[state[idx]] += 1

        for type, count in counter.items():
            result["Type"].append(type)
            result["Count"].append(count)

        head += step
    return pd.DataFrame(result)


def compute_duration(state, raw):
    state = np.array(list(map(encode, state)))
    flicker_bout_start = np.where(np.diff(state, prepend=[-1]) != 0)[0]
    result = {"Type": [], "Duration": []}
    for id1, id2 in zip(flicker_bout_start[:-1], flicker_bout_start[1:]):
        result["Type"].append(state[id1])
        result["Duration"].append((id2 - id1) * 1 / 15)

    return pd.DataFrame(result)


def stat_eval(func):
    dfs = []
    for file in glob.glob(FLICKER_DIR + "*"):
        f1 = pd.read_csv(file)

        channels = ""
        for column in f1.columns:
            if "CA1" in column:
                channels = re.search(r"-\d+-\d+", column).group()
                print(f"Found channels: {channels} in column: {column}")
                break
        if not channels:
            print(f"Channel not found for {file}")

        state = f1[
            [f"CA1{channels}-flicker-state", f"CA1{channels}-surrounding-state"]
        ].apply(lambda x: f"{x[0]}-{x[1]}" if x[0] != -1 else "None", axis=1)
        tempd = func(state, f1)
        animal = re.search(r"(CAF|KDR)\d+", file).group()
        tempd["Animal"] = animal
        dfs.append(tempd)

    return pd.concat(dfs)
