"""
modules/emotion_trainer.py
───────────────────────────
Train the FER2013 emotion CNN model.
Downloads FER2013 via kagglehub, trains a CNN, saves as
models/face_sentiment_model.h5

Run via:  python main.py --mode train
"""

import os


def train_emotion_model():
    try:
        import kagglehub
        import tensorflow as tf
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            Conv2D, MaxPooling2D, Flatten, Dense,
            Dropout, BatchNormalization
        )
        from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Install with: pip install kagglehub tensorflow")
        return

    os.makedirs("models", exist_ok=True)

    # ── 1. Download dataset ──────────────────────────────────────────────────
    print("📦 Downloading FER2013 dataset via kagglehub …")
    path = kagglehub.dataset_download("msambare/fer2013")
    print(f"✅ Dataset at: {path}")

    train_dir = os.path.join(path, "train")
    test_dir  = os.path.join(path, "test")

    # ── 2. Data generators ───────────────────────────────────────────────────
    print("📂 Building data generators …")
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    BATCH = 64
    IMG   = (48, 48)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=IMG, batch_size=BATCH,
        color_mode="grayscale", class_mode="categorical"
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir, target_size=IMG, batch_size=BATCH,
        color_mode="grayscale", class_mode="categorical"
    )

    num_classes = train_gen.num_classes
    print(f"   Classes detected: {list(train_gen.class_indices.keys())}")

    # ── 3. Build model ───────────────────────────────────────────────────────
    print("🧠 Building CNN model …")
    model = Sequential([
        # Block 1
        Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=(48, 48, 1)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 2
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 3
        Conv2D(256, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Dense head
        Flatten(),
        Dense(512, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # ── 4. Train ─────────────────────────────────────────────────────────────
    print("🚀 Training … (this may take a while)")
    callbacks = [
        ModelCheckpoint(
            "models/face_sentiment_model.h5",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    ]

    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=30,
        callbacks=callbacks,
    )

    # ── 5. Save ──────────────────────────────────────────────────────────────
    model.save("models/face_sentiment_model.h5")
    final_acc = max(history.history.get("val_accuracy", [0]))
    print(f"\n✅ Training complete. Best val accuracy: {final_acc*100:.1f}%")
    print("   Model saved → models/face_sentiment_model.h5")
    print("   Now run:  python main.py --mode live")
  
