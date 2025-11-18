import tensorflow as tf
from keras import layers, models

def conv_block(inputs, filters):
    """
    基礎卷積塊：執行兩次卷積。
    inputs: 輸入的張量
    filters: 卷積核的數量 (特徵深度)
    """
    # 第一次卷積
    x = layers.Conv2D(filters, (3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x) # 加速訓練，穩定收斂
    x = layers.Activation("relu")(x)   # 引入非線性

    # 第二次卷積
    x = layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    return x

def encoder_block(inputs, filters):
    """
    編碼器：提取特徵 (conv) -> 降採樣 (pool)
    """
    x = conv_block(inputs, filters)   # 提取特徵
    p = layers.MaxPooling2D((2, 2))(x) # 圖片變小一半 (降採樣)
    # x 透過 skip connection 傳給編碼器
    # p 傳給下一層
    return x, p  


def decoder_block(inputs, skip_features, filters):
    """
    解碼器：上採樣 (transpose conv) -> 拼接 (concat) -> 提取特徵 (conv)
    """
    # 1. 上採樣 (圖片放大兩倍)
    x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same")(inputs)
    
    # 2. 跳躍連接 (Skip Connection)
    # 與編碼器保存的高解析度特徵 (skip_features) 合併
    x = layers.concatenate([x, skip_features])
    
    # 3. 一次卷積融合特徵
    x = conv_block(x, filters)
    return x

def build_audio_unet(input_shape):
    """
    組裝完整的 U-Net 模型
    input_shape: 例如 (512, 128, 1) -> (頻率, 時間, 1)
    """
    inputs = layers.Input(input_shape)

    # --- [編碼器路徑 (Encoder)]：特徵越來越多，圖片越來越小 ---
    # 假設輸入是 (512, 128, 1)
    s1, p1 = encoder_block(inputs, 16)  # -> (256, 64, 16)
    s2, p2 = encoder_block(p1, 32)      # -> (128, 32, 32)
    s3, p3 = encoder_block(p2, 64)      # -> (64, 16, 64)
    s4, p4 = encoder_block(p3, 128)     # -> (32, 8, 128)

    # --- [瓶頸層 (Bottleneck)] ---
    b1 = conv_block(p4, 256)            # -> (16, 4, 256)

    # --- [解碼器路徑 (Decoder)]：圖片越來越大，還原細節 ---
    d1 = decoder_block(b1, s4, 128)     # 回到 (32, 8, 128)
    d2 = decoder_block(d1, s3, 64)      # 回到 (64, 16, 64)
    d3 = decoder_block(d2, s2, 32)      # 回到 (128, 32, 32)
    d4 = decoder_block(d3, s1, 16)      # 回到 (256, 64, 16)

    # --- [輸出層 (Output)] ---
    # 目標：輸出 2 張遮罩：一張給旋律，一張給伴奏
    # 使用 Sigmoid 讓每個像素的值介於 0~1 之間 (代表能量保留的比例)
    outputs = layers.Conv2D(2, (1, 1), padding="same", activation="sigmoid")(d4)

    # 建立模型
    model = models.Model(inputs, outputs, name="Audio_UNet")
    return model

# 假設你在 model.py 裡寫了上面的程式碼
# from model import build_audio_unet (如果你分開檔案的話)

# 1. 定義輸入尺寸
# 重要：U-Net 喜歡「2 的次方」，例如 256, 512, 1024。
# 如果你的 STFT 是 1025 (librosa 預設)，你可能要切掉一行變成 1024。
INPUT_SHAPE = (512, 256, 1)  # (頻率軸, 時間軸, 單聲道)

# 2. 建立模型
model = build_audio_unet(INPUT_SHAPE)

# 3. 查看架構 (這步很重要，檢查有沒有報錯)
model.summary()

# 4. 編譯模型 (設定學習方式)
# Optimizer: Adam 是最常用的
# Loss: 'mae' (平均絕對誤差) 對還原頻譜圖效果不錯
model.compile(optimizer="adam", loss="mae", metrics=["mae"])

print("模型建置完成！引擎準備好了。")