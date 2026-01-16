import pandas as pd
import numpy as np
from pathlib import Path

def load_data() -> pd.DataFrame:
    df = pd.read_csv("sp500_data.csv", skiprows=[1])
    
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    
    initial_len = len(df)
    # Forward fill then backward fill
    df = df.ffill().bfill()
    # Drop any remaining NaN rows
    df = df.dropna()
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} rows with missing values")
    
    return df

def preprocess(data, window_size=60, stride=1, train_ratio=0.6, val_ratio=0.2, save_dir=None):
    n = len(data)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))
    train_raw, val_raw, test_raw = data[:t1], data[t1:t2], data[t2:]

    mean = train_raw.mean(axis=0)
    std = np.where(train_raw.std(axis=0) == 0, 1, train_raw.std(axis=0))

    norm = lambda x: (x - mean) / std
    train, val, test = norm(train_raw), norm(val_raw), norm(test_raw)

    def make_windows(x):
        x = np.asarray(x)
        n = len(x)
        nw = (n - window_size) // stride + 1
        idx = np.arange(window_size)[None, :] + stride * np.arange(nw)[:, None]
        return x[idx]

    train_w = make_windows(train)
    val_w = make_windows(val)
    test_w = make_windows(test)

    if save_dir:
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        np.savez(
            f"{save_dir}/preprocessed.npz",
            train=train_w,
            val=val_w,
            test=test_w,
            mean=mean,
            std=std,
        )

    return train_w, val_w, test_w, {"mean": mean, "std": std}


print(preprocess(load_data(), save_dir="test"))