from pickle import TRUE
import numpy as np
import neuraltoolkit as ntk
import glob
import csv
import os
import smart_open
import boto3
import tqdm
import argparse
import fnmatch
import io
import re
import tempfile
from pathlib import Path

ANIMAL = "CAF49"
NUM_CHANNELS = 64
ERROR = True
COPY_ONLY = False
FAKE = False

FIELDS = [
    "activity",
    "sleep_state",
    "next_wake_state",
    "next_nrem_state",
    "next_rem_state",
    "last_wake_state",
    "last_nrem_state",
    "last_rem_state",
    "video_filename_ix",
    "video_frame_offset",
    "neural_filename_ix",
    "neural_offset",
]


def get_remote_files(prefix: str):
    try:
        response = client.list_objects_v2(Bucket="hengenlab", Prefix=prefix)["Contents"]
    except:
        response = []
    return ntk.ntk_videos.natural_sort([file["Key"] for file in response])


def load_raw_binary_gain_chmap(name, number_of_channels, hstype, nprobes=1, t_only=0):
    """
    load ecube data and multiply gain and apply channel mapping
    load_raw_binary_gain_chmap(name, number_of_channels, hstype)
    hstype : 'hs64', 'Si_64_KS_chmap', 'Si_64_KT_T1_K2_chmap' and linear
    nprobes : Number of probes (default 1)
    t_only  : if t_only=1, just return tr, timestamp
              (Default 0, returns timestamp and data)
    returns first timestamp and data
    """

    from neuraltoolkit import ntk_channelmap as ntkc

    if isinstance(hstype, str):
        hstype = [hstype]

    assert len(hstype) == nprobes, "length of hstype not same as nprobes"

    # constants
    gain = np.float64(0.19073486328125)

    f = name
    tr = np.frombuffer(f, dtype=np.uint64, count=1)
    if t_only:
        return tr

    dr = np.frombuffer(f, dtype=np.int16, count=-1)
    length = np.int64(np.size(dr) / number_of_channels)
    drr = np.reshape(dr, [number_of_channels, length], order="F")
    dg = drr * gain
    dgc = ntkc.channel_map_data(dg, number_of_channels, hstype, nprobes)
    return tr, dgc


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


def arg_parse():
    parser = argparse.ArgumentParser(
        prog="sample",
        description="Sample and align data from neural recordings and sleep labels",
    )
    parser.add_argument("--animal", required=True, type=str)
    parser.add_argument("--num-channels", required=True, type=int)
    parser.add_argument("--error", required=False, type=bool)
    parser.add_argument("--sleep-pattern", required=False, type=str)
    parser.add_argument("--copy-only", required=False, type=bool)
    parser.add_argument("--fake", required=False, type=bool)

    return vars(parser.parse_args())


def get_files(dir: str, pattern: str, client: boto3.client) -> list:
    files = ntk.ntk_videos.natural_sort(
        [
            obj["Key"]
            for obj in client.list_objects_v2(Bucket="hengenlab", Prefix=dir)[
                "Contents"
            ]
        ]
    )
    files = [line for line in files if fnmatch.fnmatch(line, pattern)]
    if not files:
        raise Exception(f"Files in {dir} that matches {pattern} are not found")
    else:
        return files


def data_sync(client, error=True, fake=False):
    neural_files = get_files(NF_DIR, NF_PATTERN, client)
    if not fake:
        label_files = get_files(SS_DIR, SS_PATTERN, client)
        sleep_labels = np.concatenate(
            [
                np.load(
                    io.BytesIO(
                        client.get_object(Bucket="hengenlab", Key=file)["Body"].read()
                    )
                )
                for file in label_files
            ]
        )
        sleep_labels[sleep_labels == 5] = 1
        sleep_labels[sleep_labels == 4] = 2

        print(
            f"Found {len(label_files)} label files and {len(neural_files)} neural files"
        )
        print(f"Found total {len(sleep_labels)} sleep labels")
    else:
        sleep_labels = -np.ones((3600 * 6))

    timestamp_gap = 5 * 60 * 10**9

    if error and abs(sleep_labels.shape[0] - 21600) > 900:
        raise Exception(
            f"Missing sleep labels (available size {sleep_labels.shape[0] // 900})"
        )

    with smart_open.open(
        OUTPUT_DIR + OUTPUT_FILENAME + ".csv", "w", transport_params=dict(client=client)
    ) as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        offset_tracker = 0
        offset_gap = 1 / 15 * FS

        print("S3 client connected...start processing...")

        for neural_file_ix, n_file in tqdm.tqdm(enumerate(neural_files)):
            print(f" Loading {n_file}")
            current_timestamp = load_raw_binary_gain_chmap(
                client.get_object(Bucket="hengenlab", Key=n_file)["Body"].read(),
                NUM_CHANNELS,
                "hs64",
                t_only=True,
            )[0]
            current_file_size = (
                (client.get_object(Bucket="hengenlab", Key=n_file)["ContentLength"] - 8)
                / NUM_CHANNELS
                / 2
            )

            if neural_file_ix == 0:
                last_timestamp = current_timestamp
                last_file_size = (
                    (
                        client.get_object(Bucket="hengenlab", Key=n_file)[
                            "ContentLength"
                        ]
                        - 8
                    )
                    / NUM_CHANNELS
                    / 2
                )

            else:
                expected_timestamp = last_timestamp + timestamp_gap
                if error and current_timestamp - expected_timestamp > 10**10:
                    raise Exception(
                        f"Inconsistent timestamp: last {last_timestamp} current {current_timestamp}"
                    )

                if error and (
                    last_file_size / FS * 10**9 != current_timestamp - last_timestamp
                ):
                    raise Exception(f"Inconsistent file size {last_file_size}")

            offset_checkpoint = offset_tracker
            ss_ix = 0
            print(f"Current time {offset_tracker // FS} secs")
            while (
                offset_tracker - offset_checkpoint <= current_file_size
                and ss_ix < sleep_labels.shape[0]
            ):
                entry = {
                    "activity": -1,
                    "sleep_state": int(sleep_labels[ss_ix]),
                    "next_wake_state": -1,
                    "next_nrem_state": -1,
                    "next_rem_state": -1,
                    "last_wake_state": -1,
                    "last_nrem_state": -1,
                    "last_rem_state": -1,
                    "video_filename_ix": -1,
                    "video_frame_offset": -1,
                    "neural_filename_ix": neural_file_ix,
                    "neural_offset": round(offset_tracker - offset_checkpoint),
                }
                w.writerow(entry)
                offset_tracker += offset_gap
                ss_ix = round(offset_tracker) // FS // 4

            last_timestamp = current_timestamp
            last_file_size = current_file_size
            if ss_ix >= sleep_labels.shape[0]:
                break


def save_npz(client):
    with smart_open.open(
        OUTPUT_DIR + OUTPUT_FILENAME + ".csv",
        "r",
        transport_params=dict(client=client),
    ) as f:
        reader = csv.reader(f)
        next(reader, None)
        videofiles = np.array(
            [
                re.search("Video/(\S+.mp4)", file).groups()[0]
                for file in get_remote_files(f"{ANIMAL}/Video/")
            ]
        )
        if videofiles.shape[0] == 0:
            videofiles = np.array(["dummy"])
        neuralfiles = np.array(
            [
                re.search("Headstages\S+", file).group(0)
                for file in get_files(dir=NF_DIR, pattern=NF_PATTERN, client=client)
            ]
        )
        labels_matrix = np.array(
            [tuple(line) for line in reader],
            dtype=[
                ("activity", int),
                ("sleep_state", int),
                ("next_wake_state", int),
                ("next_nrem_state", int),
                ("next_rem_state", int),
                ("last_wake_state", int),
                ("last_nrem_state", int),
                ("last_rem_state", int),
                ("video_filename_ix", int),
                ("video_frame_offset", int),
                ("neural_filename_ix", int),
                ("neural_offset", int),
            ],
        )
        with smart_open.open(
            OUTPUT_DIR + OUTPUT_FILENAME + ".npz",
            "wb",
            transport_params=dict(client=client),
        ) as of:
            np.savez(
                of,
                labels_matrix=labels_matrix,
                video_files=videofiles,
                neural_files=neuralfiles,
            )


if __name__ == "__main__":
    vars = arg_parse()
    ANIMAL = vars["animal"]
    NUM_CHANNELS = vars["num_channels"]
    if vars["sleep_pattern"] is not None:
        SS_PATTERN = vars["sleep_pattern"]
    if vars["error"] is not None:
        ERROR = vars["sleep_pattern"]
    if vars["copy_only"] is not None:
        COPY_ONLY = vars["copy_only"]
    if vars["fake"] is not None:
        FAKE = vars["fake"]

    FS = 25000
    NF_DIR = f"{ANIMAL}/Neural_Data/"
    SS_DIR = f"{ANIMAL}/SleepState/"
    OUTPUT_DIR = f"s3://hengenlab/{ANIMAL}/Labels/"
    OUTPUT_FILENAME = f"labels_sleepstate_v2.1_{ANIMAL}"
    NF_PATTERN = r"*Headstages*.bin"
    SS_PATTERN = r"*SleepStates*.npy"

    client = get_s3_client()
    if not COPY_ONLY:
        data_sync(client, error=ERROR, fake=FAKE)
    save_npz(client)
