import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Layer
import tensorflow as tf
import tensorflow.keras.backend as K  # 导入 Keras backend

# ----------------------- 数据加载与预处理 -----------------------

# 文件路径
file_path = './training.1600000.processed.noemoticon.csv'

# 读取 CSV 文件，指定编码为 'ISO-8859-1'
df = pd.read_csv(file_path, encoding='ISO-8859-1', header=None)

# 定义列名
df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

# 选择 sentiment 和 text 列
df = df[['sentiment', 'text']]

# 将情感标签转换为 0 和 1（0 为负面，1 为正面）
df['sentiment'] = df['sentiment'].replace({4: 1, 0: 0})


# ----------------------- 文本预处理 -----------------------

# 定义文本预处理函数
def preprocess_text(text):
    text = text.lower()  # 转为小写
    text = re.sub(r'\d+', '', text)  # 移除数字
    text = re.sub(r'[^\w\s]', '', text)  # 移除标点符号
    text = re.sub(r'\s+', ' ', text).strip()  # 移除多余的空格
    return text


# 应用文本预处理
df['text'] = df['text'].apply(preprocess_text)

# ----------------------- 数据分割 -----------------------

# 打乱数据
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 创建 5 个数据子集（模拟不同节点的数据）
node_data = []
chunk_size = len(df) // 5

for i in range(5):
    start_idx = i * chunk_size
    end_idx = (i + 1) * chunk_size if i != 4 else len(df)  # 最后一个子集包含剩余数据
    node_data.append(df[start_idx:end_idx])

# ----------------------- 文本向量化 -----------------------

# 初始化 Tokenizer，限制最多使用 10000 个单词
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df['text'])  # 基于文本数据训练 Tokenizer

# ----------------------- 数据划分：训练、验证、测试 -----------------------

node_splits = {}
global_test_texts = []
global_test_labels = []

for i, data in enumerate(node_data):
    # 每个节点的数据划分为训练集、验证集和测试集
    node_train_val, node_global_test = train_test_split(
        data, test_size=0.2, random_state=42  # 20% 数据用于全局测试集
    )

    # 收集全局测试集数据
    global_test_texts.extend(node_global_test['text'].tolist())
    global_test_labels.extend(node_global_test['sentiment'].tolist())

    # 将剩余数据划分为训练集、验证集和节点测试集
    node_train, node_test = train_test_split(
        node_train_val, test_size=0.2, random_state=42  # 20% 数据用于节点测试集
    )
    X_train, X_val, y_train, y_val = train_test_split(
        node_train['text'], node_train['sentiment'], test_size=0.2, random_state=42
    )

    # 保存数据
    node_splits[f"Node_{i + 1}"] = {
        "X_train": pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=100),
        "X_val": pad_sequences(tokenizer.texts_to_sequences(X_val), maxlen=100),
        "X_test": pad_sequences(tokenizer.texts_to_sequences(node_test['text']), maxlen=100),
        "y_train": y_train.values,
        "y_val": y_val.values,
        "y_test": node_test['sentiment'].values
    }


# ----------------------- 自定义自注意力层 -----------------------

class SelfAttention(Layer):
    def __init__(self):
        super(SelfAttention, self).__init__()

    def build(self, input_shape):
        # 设置可训练的权重参数
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(input_shape[-1],), initializer="zeros", trainable=True)

    def call(self, inputs):
        # Query矩阵
        Q = K.dot(inputs, self.W) + self.b

        # 对Q进行reshape，展平为(batch_size, seq_len, embedding_dim)
        Q = K.reshape(Q, (-1, K.shape(Q)[1], K.shape(Q)[2]))

        # 使用 tf.transpose 来替代 K.transpose
        Q_T = tf.transpose(Q, perm=[0, 2, 1])  # 转置 Q，形状变为 (batch_size, embedding_dim, seq_len)

        # 计算注意力矩阵 A
        A = tf.nn.softmax(K.batch_dot(Q, Q_T))  # 使用 batch_dot 来避免维度问题

        # 输出加权和
        output = K.batch_dot(A, inputs)

        return output

    def compute_output_shape(self, input_shape):
        # 输出形状与输入形状相同
        return input_shape


# ----------------------- 创建 Bi-LSTM + CNN + 自注意力模型 -----------------------

def create_bilstm_cnn_attention_model(input_dim, output_dim, input_length):
    model = Sequential([
        # Embedding Layer
        Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length),

        # Bi-LSTM Layer (双向LSTM)
        Bidirectional(LSTM(128, return_sequences=True)),

        # 自注意力机制
        SelfAttention(),

        # CNN Layer (卷积层提取深层次特征)
        Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),  # 池化操作降低维度

        # 另一层卷积和池化（可以尝试多个卷积层）
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),

        # Global Max Pooling (全局最大池化，减少维度)
        GlobalMaxPooling1D(),

        # 全连接层进行分类
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # 输出层，sigmoid用于二分类
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ----------------------- 权重平均函数 -----------------------

def average_weights(local_weights, neighbor_weights):
    """
    计算本地模型的权重与邻居模型权重的平均值
    :param local_weights: 当前节点模型的权重
    :param neighbor_weights: 邻居节点模型的权重
    :return: 平均后的权重
    """
    for layer in range(len(local_weights)):
        local_weights[layer] = 0.5 * local_weights[layer] + 0.5 * neighbor_weights[layer]
    return local_weights


# ----------------------- 训练与权重共享 -----------------------

# 初始化节点模型
trained_models = {}
for node_name, splits in node_splits.items():
    trained_models[node_name] = create_bilstm_cnn_attention_model(input_dim=10000, output_dim=128, input_length=100)

# 训练与权重共享
epochs = 5  # 训练 5 轮
trained_histories = {}  # 保存每个节点的训练历史

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs} starting...")

    # 每个节点训练 1 轮
    for node_name, splits in node_splits.items():
        print(f"{node_name} training starts...")

        # 训练并保存历史
        history = trained_models[node_name].fit(
            splits['X_train'], splits['y_train'],
            validation_data=(splits['X_val'], splits['y_val']),
            epochs=1,  # 每个节点训练 1 轮
            batch_size=32,
            verbose=1
        )

        if node_name not in trained_histories:
            trained_histories[node_name] = {
                "loss": [],
                "accuracy": [],
                "val_loss": [],
                "val_accuracy": []
            }

        trained_histories[node_name]["loss"].extend(history.history.get("loss", []))
        trained_histories[node_name]["accuracy"].extend(history.history.get("accuracy", []))
        trained_histories[node_name]["val_loss"].extend(history.history.get("val_loss", []))
        trained_histories[node_name]["val_accuracy"].extend(history.history.get("val_accuracy", []))

    # 权重共享：更新每个节点的权重
    print(f"Epoch {epoch + 1}: Weight sharing starts...")
    updated_weights_dict = {}

    for node_name, model in trained_models.items():
        neighbors = [trained_models[neighbor_name] for neighbor_name in trained_models if neighbor_name != node_name]
        neighbor_weights = [
            np.mean([neighbor.get_weights()[layer] for neighbor in neighbors], axis=0)
            for layer in range(len(model.get_weights()))
        ]
        updated_weights_dict[node_name] = average_weights(model.get_weights(), neighbor_weights)

    # 更新权重
    for node_name, updated_weights in updated_weights_dict.items():
        trained_models[node_name].set_weights(updated_weights)

    print(f"Epoch {epoch + 1}: Weight sharing completed.")
    print("-" * 40)

print("Training process completed.")

# ----------------------- 在节点本地测试集上评估 -----------------------

print("Performance on each node's test data:")

for node_name, splits in node_splits.items():
    X_test = splits['X_test']
    y_test = splits['y_test']

    predictions = trained_models[node_name].predict(X_test)
    predictions = (predictions > 0.5).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    print(f"{node_name} - Test Accuracy: {accuracy:.4f}")

# ----------------------- 全局模型评估 -----------------------

# 组织全局测试集
global_test_df = pd.DataFrame({
    "text": global_test_texts,
    "sentiment": global_test_labels
})

X_global_test = pad_sequences(tokenizer.texts_to_sequences(global_test_df['text']), maxlen=100)
y_global_test = global_test_df['sentiment'].values

# 获取所有节点的平均权重
global_weights = [
    np.mean([model.get_weights()[layer] for model in trained_models.values()], axis=0)
    for layer in range(len(list(trained_models.values())[0].get_weights()))
]

# 创建全局模型并设置权重
global_model = create_bilstm_cnn_attention_model(input_dim=10000, output_dim=128, input_length=100)
global_model.build(input_shape=(None, 100))
global_model.set_weights(global_weights)

# 评估全局模型
global_predictions = global_model.predict(X_global_test)
global_predictions = (global_predictions > 0.5).astype(int)

global_accuracy = accuracy_score(y_global_test, global_predictions)
print(f"Global Model - Test Accuracy: {global_accuracy:.4f}")

# ----------------------- 可视化 -----------------------

# 绘制每个节点的训练过程（损失与准确度）
for node_name, history in trained_histories.items():
    # 绘制训练和验证损失曲线
    plt.plot(history["loss"], label=f"{node_name} - Train Loss")
    plt.plot(history["val_loss"], label=f"{node_name} - Val Loss")

plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('loss_curve_bilstm_cnn_attention.png')
plt.clf()

# 绘制每个节点的训练过程（准确度）
for node_name, history in trained_histories.items():
    # 绘制训练和验证准确度曲线
    plt.plot(history["accuracy"], label=f"{node_name} - Train Accuracy")
    plt.plot(history["val_accuracy"], label=f"{node_name} - Val Accuracy")

plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('accuracy_curve_bilstm_cnn_attention.png')
plt.clf()
