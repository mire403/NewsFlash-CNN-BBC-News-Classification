import csv
import time
import numpy as np
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from matplotlib import pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =========================
# 0. 日志系统（print + txt）
# =========================
LOG_FILE = "experiment_report.txt"

def log(msg=""):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

open(LOG_FILE, "w").close()  # 清空旧日志

# =========================
# 1. 参数配置
# =========================
nltk.data.path.append("./nltk_data")

VOCAB_SIZE = 5000
MAX_LENGTH = 200
EMBEDDING_DIM = 64
TRAIN_RATIO = 0.8
EPOCHS = 10
OOV_TOKEN = "<OOV>"

# =========================
# 2. 停用词加载
# =========================
STOPWORDS = set(stopwords.words("english"))
log(f"[INFO] Stopwords loaded: {len(STOPWORDS)}")

# =========================
# 3. 数据读取与预处理
# =========================
articles, labels = [], []

with open("bbc-text.csv", "r", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for label, text in reader:
        for w in STOPWORDS:
            text = text.replace(f" {w} ", " ")
        articles.append(text)
        labels.append(label)

log(f"[INFO] Total samples: {len(articles)}")

# =========================
# 4. 数据分布统计（文字 + 图）
# =========================
counter = Counter(labels)
log("\n[DATASET DISTRIBUTION]")
for k, v in counter.items():
    log(f"{k:<15}: {v}")

plt.figure()
plt.bar(counter.keys(), counter.values())
plt.title("Label Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("label_distribution.png")
plt.close()

# =========================
# 5. 文本序列化
# =========================
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(articles)

sequences = tokenizer.texts_to_sequences(articles)
padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding="post")

lengths = [len(seq) for seq in sequences]

log("\n[VOCAB INFO]")
log(f"Actual vocabulary size: {len(tokenizer.word_index)}")
log(f"Average text length: {np.mean(lengths):.2f}")

plt.figure()
plt.hist(lengths, bins=50)
plt.title("Text Length Distribution")
plt.xlabel("Length")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("length_distribution.png")
plt.close()

# =========================
# 6. 标签编码
# =========================
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)
label_seq = np.array(label_tokenizer.texts_to_sequences(labels)) - 1
num_classes = len(label_tokenizer.word_index)

log("\n[LABEL MAPPING]")
for label, idx in label_tokenizer.word_index.items():
    log(f"{label:<15} -> {idx - 1}")

# =========================
# 7. 划分数据集
# =========================
train_size = int(len(articles) * TRAIN_RATIO)

X_train, X_val = padded[:train_size], padded[train_size:]
y_train, y_val = label_seq[:train_size], label_seq[train_size:]

log("\n[DATA SPLIT]")
log(f"Train samples: {len(X_train)}")
log(f"Validation samples: {len(X_val)}")

# =========================
# 8. 模型构建
# =========================
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM),
    tf.keras.layers.Conv1D(256, 3, activation="relu", padding="same"),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(EMBEDDING_DIM, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

log("\n[MODEL SUMMARY]")
model.summary(print_fn=log)

# =========================
# 9. 训练
# =========================
log("\n[TRAINING START]")
start = time.time()

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    verbose=2
)

log(f"[TRAINING END] Time used: {time.time() - start:.2f}s")

# =========================
# 10. 训练曲线
# =========================
for metric in ["accuracy", "loss"]:
    plt.figure()
    plt.plot(history.history[metric], label="train")
    plt.plot(history.history["val_" + metric], label="val")
    plt.title(metric.capitalize())
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{metric}.png")
    plt.close()

# =========================
# 11. 混淆矩阵
# =========================
y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)

cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=label_tokenizer.word_index.keys())
disp.plot(xticks_rotation=45)
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# =========================
# 12. 单条预测示例（写入 txt）
# =========================
def predict_text(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LENGTH, padding="post")
    probs = model.predict(pad, verbose=0)[0]
    idx = np.argmax(probs)
    label = list(label_tokenizer.word_index.keys())[idx]
    return label, probs[idx]

log("\n[SAMPLE PREDICTION]")
sample = articles[5]
pred, conf = predict_text(sample)
log(f"Text snippet: {sample[:120]}...")
log(f"Predicted label: {pred}")
log(f"Confidence: {conf:.4f}")
