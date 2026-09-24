"""
Data pipeline for the 3-class rain intensity CNN

Functions:
   1. generate_balanced_3class_csv: builds a balanced 70/15/15 CSV from
      the binary CSV, keeping the 11 segments of each audio in the same split
   2. build_datasets: reads the split CSV and builds three
      tf.data.Dataset objects for model.fit(), with optional
      SpecAugment, focused Mixup and one-hot labels

Per-image preprocessing:
   1. Reads the PNG from disk
   2. Decodes it with N_CHANNELS channels
   3. Resizes from 512x512 to 256x256 with bilinear interpolation
   4. Applies per-image standardization
   5. Train only: SpecAugment time/frequency masking, brightness and contrast
   6. Train only: focused Mixup between light and moderate

"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

# General settings
IMAGE_SIZE     = (256, 256)          
N_CHANNELS     = 1                    
N_CLASSES      = 3                  
BATCH_SIZE     = 64                  
SHUFFLE_BUFFER = 2048                  
SEED           = 42
AUTOTUNE       = tf.data.AUTOTUNE

# SpecAugment hyperparameters
SPECAUG_FREQ_MASK = 30     
SPECAUG_TIME_MASK = 40     
SPECAUG_NUM_MASKS = 2     
SPECAUG_PROB      = 0.8    
SPECAUG_BRIGHTNESS = 0.25   
SPECAUG_CONTRAST_LO = 0.90  
SPECAUG_CONTRAST_HI = 1.10  

# Focused Mixup hyperparameters - light/moderate boundary
MIXUP_ALPHA = 0.4  
MIXUP_PROB  = 0.5    

# GPU setup
def configure_gpu(verbose: bool = True) -> None:
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


# Balanced 3-class CSV generation
def generate_balanced_3class_csv(
    input_csv,
    output_csv,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    test_frac:  float = 0.15,
    seed:       int   = SEED,
    verbose:    bool  = True,) -> pd.DataFrame:
    """
    Class mapping:
        light    -> light    (label 0)
        moderate -> moderate (label 1)
        heavy    -> intense  (label 2)
        violent  -> intense  (label 2)
        no-rain  -> discarded
    """
    input_csv  = Path(input_csv)
    output_csv = Path(output_csv)

    if not input_csv.is_file():
        raise FileNotFoundError(f"CSV não encontrado: {input_csv.resolve()}")

    df = pd.read_csv(input_csv)
    rain_df = df[df["class_name"] != "no-rain"].copy()

    name_to_3class = {
        "light":    "light",
        "moderate": "moderate",
        "heavy":    "intense",
        "violent":  "intense",}
    
    rain_df["class_name"] = rain_df["class_name"].map(name_to_3class)
    label_map = {"light": 0, "moderate": 1, "intense": 2}

    audios_per_class = {
        cls: sorted(rain_df.loc[rain_df["class_name"] == cls, "audio_id"]
                            .unique().tolist())
        for cls in label_map}

    if verbose:
        print("\n[GEN] Áudios únicos por classe (antes do balanceamento):")
        for cls, audios in audios_per_class.items():
            n_segs = int((rain_df["class_name"] == cls).sum())
            print(f"       - {cls:>8}: {len(audios):>4} áudios | "
                  f"{n_segs:>5} segmentos")

    min_n = min(len(a) for a in audios_per_class.values())
    rng = np.random.RandomState(seed)
    balanced = {}
    for cls, audios in audios_per_class.items():
        shuffled = list(audios)
        rng.shuffle(shuffled)
        balanced[cls] = shuffled[:min_n]

    n_train = int(min_n * train_frac)
    n_val   = int(min_n * val_frac)

    audio_to_split = {}
    for cls, audios in balanced.items():
        for a in audios[:n_train]:                  audio_to_split[a] = "train"
        for a in audios[n_train:n_train + n_val]:   audio_to_split[a] = "val"
        for a in audios[n_train + n_val:]:          audio_to_split[a] = "test"

    kept = set(audio_to_split)
    new_df = rain_df[rain_df["audio_id"].isin(kept)].copy()
    new_df["split"] = new_df["audio_id"].map(audio_to_split)
    new_df["label"] = new_df["class_name"].map(label_map).astype(int)
    new_df = new_df[["path", "label", "class_name", "audio_id", "split"]] \
                .reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(output_csv, index=False)

    if verbose:
        print(f"\n[GEN] CSV salvo em: {output_csv.resolve()}")
        print(f"[GEN] Total: {len(new_df)} segmentos | "
              f"{new_df['audio_id'].nunique()} áudios")
        print("\n[GEN] Distribuição final (áudios | segmentos):")
        for split in ["train", "val", "test"]:
            print(f"\n   {split}:")
            sdf = new_df[new_df["split"] == split]
            for cls in label_map:
                n_aud  = int(sdf.loc[sdf["class_name"] == cls,
                                      "audio_id"].nunique())
                n_segs = int((sdf["class_name"] == cls).sum())
                print(f"     {cls:>8}: {n_aud:>3} áudios | "
                      f"{n_segs:>4} segmentos")

        leaks = []
        for s1, s2 in [("train", "val"), ("train", "test"), ("val", "test")]:
            a1 = set(new_df.loc[new_df["split"] == s1, "audio_id"])
            a2 = set(new_df.loc[new_df["split"] == s2, "audio_id"])
            if a1 & a2:
                leaks.append((s1, s2, len(a1 & a2)))
        if leaks:
            print(f"\n[GEN] ERRO: vazamento de áudio entre splits: {leaks}")
        else:
            print(f"\n[GEN] OK: nenhum vazamento de áudio entre splits.")

    return new_df

# Image loading and preprocessing
def load_and_preprocess_image(path: tf.Tensor, label):
# Reads the PNG, decodes it, resizes it to IMAGE_SIZE and applies per-image standardization
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_png(img_bytes, channels=N_CHANNELS)
    img = tf.image.resize(img, IMAGE_SIZE, method="bilinear")
    img = tf.image.per_image_standardization(img)  # zero mean, unit variance
    return img, label


# SpecAugment (time/frequency masking)
def _apply_one_mask(image: tf.Tensor, max_width: int, axis: int) -> tf.Tensor:
    """
    Zeroes a random band along the given axis
      axis=0  -> horizontal band (frequency mask)
      axis=1  -> vertical band   (time mask)
    """
    img_h, img_w = IMAGE_SIZE
    width = tf.random.uniform([], 1, max_width + 1, dtype=tf.int32)

    if axis == 0:
        start = tf.random.uniform([], 0, img_h - width, dtype=tf.int32)
        mask = tf.concat([
            tf.ones ([start,             img_w, N_CHANNELS]),
            tf.zeros([width,             img_w, N_CHANNELS]),
            tf.ones ([img_h - start - width, img_w, N_CHANNELS]),], axis=0)
    else:
        start = tf.random.uniform([], 0, img_w - width, dtype=tf.int32)
        mask = tf.concat([
            tf.ones ([img_h, start,             N_CHANNELS]),
            tf.zeros([img_h, width,             N_CHANNELS]),
            tf.ones ([img_h, img_w - start - width, N_CHANNELS]),], axis=1)

    return image * mask


def spec_augment(image: tf.Tensor, label):
    """
    Applies SPECAUG_NUM_MASKS frequency masks, SPECAUG_NUM_MASKS time masks,
    random brightness and random contrast, with probability SPECAUG_PROB
    Train only; the decision is made per sample
    """
    def augment():
        img = image
        for _ in range(SPECAUG_NUM_MASKS):
            img = _apply_one_mask(img, SPECAUG_FREQ_MASK, axis=0)
        for _ in range(SPECAUG_NUM_MASKS):
            img = _apply_one_mask(img, SPECAUG_TIME_MASK, axis=1)
        img = tf.image.random_brightness(img, max_delta=SPECAUG_BRIGHTNESS)
        img = tf.image.random_contrast(img, lower=SPECAUG_CONTRAST_LO, upper=SPECAUG_CONTRAST_HI)
        return img

    apply = tf.random.uniform([], 0.0, 1.0) < SPECAUG_PROB
    image = tf.cond(apply, augment, lambda: image)
    return image, label

# Focused Mixup (light/moderate boundary)
def focused_mixup(images: tf.Tensor, labels: tf.Tensor):
    """
    Mixup restricted to classes 0 (light) and 1 (moderate)
    - Pairs where both samples are light/moderate: mixed with lam ~ Beta(α, α)
    """
    def apply_mixup():
        batch_size = tf.shape(images)[0]
        gamma1 = tf.random.gamma([1], MIXUP_ALPHA)
        gamma2 = tf.random.gamma([1], MIXUP_ALPHA)
        lam = tf.reshape(gamma1 / (gamma1 + gamma2), [])

        # Shuffle the batch to form pairs
        perm = tf.random.shuffle(tf.range(batch_size))
        images_perm = tf.gather(images, perm)
        labels_perm = tf.gather(labels, perm)

        # Mix only pairs where both samples are light (0) or moderate (1)
        cls_orig = tf.argmax(labels,      axis=1)
        cls_perm = tf.argmax(labels_perm, axis=1)
        do_mix = tf.logical_and(cls_orig < 2, cls_perm < 2)
        do_mix_img = tf.cast(do_mix, tf.float32)[:, None, None, None]
        do_mix_lbl = tf.cast(do_mix, tf.float32)[:, None]

        # Apply only where do_mix is True
        mixed_images = do_mix_img * (lam * images + (1 - lam) * images_perm) \
                     + (1 - do_mix_img) * images
        mixed_labels = do_mix_lbl * (lam * labels + (1 - lam) * labels_perm) \
                     + (1 - do_mix_lbl) * labels

        return mixed_images, mixed_labels

    apply = tf.random.uniform([], 0.0, 1.0) < MIXUP_PROB
    images, labels = tf.cond(apply, apply_mixup, lambda: (images, labels))
    return images, labels

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


def make_dataset(df: pd.DataFrame,
                 shuffle: bool,
                 batch_size: int,
                 label_mode: str = "int",
                 augment: bool = False,
                 mixup: bool = False) -> tf.data.Dataset:

    if mixup and label_mode != "one_hot":
        raise ValueError("mixup=True requer label_mode='one_hot' para funcionar corretamente.")

    if shuffle:
        df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    paths  = df["path"].values
    labels = df["label"].values.astype(np.int32)

    if label_mode == "one_hot":
        labels = tf.keras.utils.to_categorical(
            labels, num_classes=N_CLASSES
        ).astype(np.float32)
    elif label_mode != "int":
        raise ValueError(f"label_mode deve ser 'int' ou 'one_hot', "
                         f"recebeu: {label_mode!r}")

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(
            buffer_size=SHUFFLE_BUFFER,
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

    if augment:
        # Per-sample SpecAugment, before batching
        ds = ds.map(spec_augment, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size)

    if mixup:
        # Per-batch Mixup, after batching
        ds = ds.map(focused_mixup, num_parallel_calls=AUTOTUNE)

    ds = ds.prefetch(AUTOTUNE)
    return ds


def build_datasets(csv_path,
                   batch_size: int = BATCH_SIZE,
                   label_mode: str = "int",
                   augment_train: bool = True,
                   mixup_train: bool = True,
                   verbose: bool = True):
# Reads the CSV and returns (train_ds, val_ds, test_ds)

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path.resolve()}")

    train_df, val_df, test_df = load_split_csv(csv_path)

    if verbose:
        print(f"\n[DATA] CSV carregado de: {csv_path.resolve()}")
        print(f"[DATA] label_mode={label_mode!r} | augment_train={augment_train} | mixup_train={mixup_train}")
        print(f"[DATA] Tamanhos (em segmentos):")
        print(f"       - train: {len(train_df):>6}")
        print(f"       - val:   {len(val_df):>6}")
        print(f"       - test:  {len(test_df):>6}")
        print(f"[DATA] Distribuição de labels no train: "
              f"{train_df['label'].value_counts().sort_index().to_dict()}")

    train_ds = make_dataset(train_df, shuffle=True,  batch_size=batch_size,
                            label_mode=label_mode, augment=augment_train, mixup=mixup_train)
    val_ds   = make_dataset(val_df,   shuffle=False, batch_size=batch_size,
                            label_mode=label_mode, augment=False, mixup=False)
    test_ds  = make_dataset(test_df,  shuffle=False, batch_size=batch_size,
                            label_mode=label_mode, augment=False, mixup=False)

    return train_ds, val_ds, test_ds


# Sanity check
def inspect_batch(ds: tf.data.Dataset, name: str = "dataset") -> None:
# Prints statistics of one batch to check shape, dtype and values before training
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
        if labels_np.ndim == 1:
            print(f"   Primeiros 10:     {labels_np[:10]}")
            unique, counts = np.unique(labels_np.astype(int), return_counts=True)
            print(f"   Distribuição:     {dict(zip(unique.tolist(), counts.tolist()))}")
        else:
            print(f"   Primeira amostra: {labels_np[0]}  (one-hot)")
            cls_idx = np.argmax(labels_np, axis=1)
            unique, counts = np.unique(cls_idx, return_counts=True)
            print(f"   Distribuição:     {dict(zip(unique.tolist(), counts.tolist()))}")
    print("-" * 60)


# Standalone execution
def main():
    parser = argparse.ArgumentParser(
        description="Constrói os tf.data.Datasets e roda sanity checks."
    )
    parser.add_argument("--csv_path",   type=Path,
                        default=Path("Split70-15-15_3class.csv"))
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--label_mode", type=str, default="int",
                        choices=["int", "one_hot"])
    parser.add_argument("--no_augment", action="store_true",
                        help="Desativa SpecAugment no train_ds")
    args = parser.parse_args()

    print("=" * 60)
    print(" CONFIGURAÇÃO DA GPU")
    print("=" * 60)
    configure_gpu()

    print("\n" + "=" * 60)
    print(" CONSTRUINDO DATASETS")
    print("=" * 60)
    train_ds, val_ds, test_ds = build_datasets(
        args.csv_path,
        batch_size=args.batch_size,
        label_mode=args.label_mode,
        augment_train=not args.no_augment,
    )

    print("\n" + "=" * 60)
    print(" SANITY CHECKS")
    print("=" * 60)
    inspect_batch(train_ds, "train_ds (com SpecAugment se augment_train=True)")
    inspect_batch(val_ds,   "val_ds (sem augmentation)")
    inspect_batch(test_ds,  "test_ds (sem augmentation)")

    print("\n[OK] Pipeline pronto para uso.\n")


if __name__ == "__main__":
    main()
