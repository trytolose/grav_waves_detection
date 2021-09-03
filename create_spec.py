from gwpy.timeseries import TimeSeries
import cv2
from joblib import Parallel, delayed
from src.transforms import min_max_scale
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def gwpy_qtransform(file_path, save_path, img_size=(256, 256)):
    x = np.load(file_path)
    x = min_max_scale(x)
    images = []
    for ii in range(3):
        strain = TimeSeries(x[ii, :], t0=0, dt=1 / 2048)

        hq = strain.q_transform(
            qrange=(4, 32), frange=(20, 1024), logf=False, whiten=False, fduration=1, tres=2 / 1000
        )
        images.append(hq)

    img = np.stack([np.array(x).T for x in images], axis=2)
    # img = np.stack([np.array(x).T[:-200] for x in images], axis=2)

    TOTAL_MIN_VAL = np.array([-90, -90, -1])  # -1
    TOTAL_MAX_VAL = np.array([4050, 4250, 150])  # 50
    min_val = -1
    max_val = 1
    img = np.clip(img, TOTAL_MIN_VAL, TOTAL_MAX_VAL)

    X_std = (img - TOTAL_MIN_VAL) / (TOTAL_MAX_VAL - TOTAL_MIN_VAL)
    img = X_std * (max_val - min_val) + min_val

    img = cv2.resize(img, img_size).astype(np.float16)
    if np.isinf(img).sum() > 0:
        print("INFINITE ALERT!!!!!!")
        print(Path(file_path).name)

    file_save_path = Path(save_path) / Path(file_path).name
    np.save(file_save_path, img)


def generate():

    INPUT_PATH = Path("/home/trytolose/rinat/kaggle/grav_waves_detection/input")
    df = pd.read_csv(INPUT_PATH / "training_labels.csv")
    files = list((INPUT_PATH / "train").rglob("*.npy"))
    FILE_PATH_DICT = {x.stem: str(x) for x in files}
    df["path"] = df["id"].apply(lambda x: FILE_PATH_DICT[x])
    Parallel(n_jobs=24)(
        delayed(gwpy_qtransform)(x, "/home/trytolose/rinat/kaggle/grav_waves_detection/input/img_fp16_256")
        for x in tqdm(df["path"].values)
    )
    print()


def get_stat(file_path, save_path, img_size=(256, 256)):
    x = np.load(file_path)
    # x = min_max_scale(x)
    images = []
    for ii in range(3):
        strain = TimeSeries(x[ii, :], t0=0, dt=1 / 2048)

        hq = strain.q_transform(qrange=(4, 32), frange=(20, 400), logf=False, whiten=True, fduration=1, tres=2 / 1000)
        images.append(hq)

    img = np.stack([np.array(x).T[:-200] for x in images], axis=2)
    img = cv2.resize(img, img_size)
    result = {}
    result["max"] = img.max(axis=(0, 1))
    result["min"] = img.min(axis=(0, 1))
    result["mean"] = img.mean(axis=(0, 1))
    result["std"] = img.max(axis=(0, 1))

    return result


generate()
