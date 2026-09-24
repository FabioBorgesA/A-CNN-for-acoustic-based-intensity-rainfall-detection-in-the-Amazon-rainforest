"""
 Data pipeline for the binary rain classification CNN (no-rain / rain)

 Reads the split CSV (columns: path, label, split) and builds three
 tf.data.Dataset objects for model.fit():
   - train_ds: shuffled, with prefetch
   - val_ds:   ordered, with prefetch
   - test_ds:  ordered, with prefetch

 Per-image preprocessing:
   1. Reads the PNG from disk
   2. Decodes it with n_channels channels
   3. Resizes from 512x512 to 256x256 with bilinear interpolation
   4. Scales pixels from [0, 255] to [0, 1]
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

# Settings
IMAGE_SIZE     = (256, 256)             
N_CHANNELS     = None
BATCH_SIZE     = 32                     
SHUFFLE_BUFFER = 2048                   
SEED           = 42
AUTOTUNE       = tf.data.AUTOTUNE       # automatic parallelism for I/O and map


def configure_gpu(verbose: bool = True) -> None:
    """
    Enables memory growth on the available GPUs so TensorFlow does not
    allocate all VRAM at once

    Must be called before any tensor operation on the GPU
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        if verbose:
            print("[GPU] Nenhuma GPU detectada — treino vai rodar na CPU.")
        return

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"[GPU] [AVISO] Não foi possível ativar memory_growth em "
                  f"{gpu.name}: {e}")

    if verbose:
        print(f"[GPU] {len(gpus)} GPU(s) detectada(s) com memory_growth ativo:")
        for gpu in gpus:
            print(f"      - {gpu.name}")



# Image loading and preprocessing
def load_and_preprocess_image(path: tf.Tensor, label: tf.Tensor, n_channels: int):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_png(img_bytes, channels=n_channels)
    img = tf.image.resize(img, IMAGE_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


# Dataset construction
def load_split_csv(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
# Reads the split CSV and returns the train, val and test DataFrames
    df = pd.read_csv(csv_path)
    required_cols = {"path", "label", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV não tem as colunas obrigatórias: {missing}")

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    return train_df, val_df, test_df


def make_dataset(df: pd.DataFrame, shuffle: bool, n_channels: int, batch_size: int) -> tf.data.Dataset:
    # Builds a tf.data.Dataset from a DataFrame
    if shuffle:
        df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    paths  = df["path"].values
    labels = df["label"].values.astype(np.float32)   # binary_crossentropy expects float
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        # Reshuffle every epoch
        ds = ds.shuffle(
            buffer_size=SHUFFLE_BUFFER,
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(lambda path, label: load_and_preprocess_image(path, label, n_channels), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)

    return ds


def build_datasets(csv_path, n_channels: int, batch_size: int = BATCH_SIZE, verbose: bool = True):
# Reads the CSV and returns (train_ds, val_ds, test_ds)
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path.resolve()}")

    train_df, val_df, test_df = load_split_csv(csv_path)

    if verbose:
        print(f"\n[DATA] CSV carregado de: {csv_path.resolve()}")
        print(f"[DATA] Tamanhos (em segmentos):")
        print(f"       - train: {len(train_df):>6}")
        print(f"       - val:   {len(val_df):>6}")
        print(f"       - test:  {len(test_df):>6}")
        print(f"[DATA] Distribuição de labels no train: "
              f"{train_df['label'].value_counts().to_dict()}")

    train_ds = make_dataset(train_df, shuffle=True, n_channels=n_channels, batch_size=batch_size)
    val_ds   = make_dataset(val_df,   shuffle=False, n_channels=n_channels, batch_size=batch_size)
    test_ds  = make_dataset(test_df,  shuffle=False, n_channels=n_channels, batch_size=batch_size)

    return train_ds, val_ds, test_ds


# Sanity check
def inspect_batch(ds: tf.data.Dataset, name: str = "dataset") -> None:
    # Prints statistics of one batch to check the pipeline before training
    print(f"\n[SANITY] Inspecionando 1 batch de '{name}'")
    print("-" * 60)
    for images, labels in ds.take(1):
        imgs_np   = images.numpy()
        labels_np = labels.numpy()
        print(f"   Imagens shape:    {tuple(images.shape)}")
        print(f"   Imagens dtype:    {images.dtype.name}")
        print(f"   Imagens min/max:  [{imgs_np.min():.4f}, {imgs_np.max():.4f}]")
        print(f"   Imagens mean:     {imgs_np.mean():.4f}")
        print(f"   Imagens std:      {imgs_np.std():.4f}")
        print(f"   Labels shape:     {tuple(labels.shape)}")
        print(f"   Labels dtype:     {labels.dtype.name}")
        print(f"   Primeiros 10:     {labels_np[:10]}")
        unique, counts = np.unique(labels_np.astype(int), return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        print(f"   Distribuição:     {dist}")
    print("-" * 60)


# Standalone execution
def main():
    parser = argparse.ArgumentParser(
        description="Constrói os tf.data.Datasets a partir do CSV e roda sanity checks."
    )
    parser.add_argument("--csv_path", type=Path, default=Path("dataset_split.csv"),
                        help="Caminho do CSV gerado pelo split_dataset.py")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help=f"Tamanho do batch (default: {BATCH_SIZE})")
    args = parser.parse_args()

    print("=" * 60)
    print(" CONFIGURAÇÃO DA GPU")
    print("=" * 60)
    configure_gpu()

    print("\n" + "=" * 60)
    print(" CONSTRUINDO DATASETS")
    print("=" * 60)
    train_ds, val_ds, test_ds = build_datasets(args.csv_path,
                                                batch_size=args.batch_size)

    print("\n" + "=" * 60)
    print(" SANITY CHECKS")
    print("=" * 60)
    inspect_batch(train_ds, "train_ds")
    inspect_batch(val_ds,   "val_ds")
    inspect_batch(test_ds,  "test_ds")

    print("\n[OK] Pipeline pronto para uso no train.py.\n")


if __name__ == "__main__":
    main()
