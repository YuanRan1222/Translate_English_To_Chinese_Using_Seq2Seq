import pandas as pd
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 关闭警告信息
import tensorflow as tf
from tensorflow.keras import layers, Model
tf.keras.backend.set_floatx("float64")  # 设置浮点位数，避免警告信息


def allow_memory_growth():  # 设置GPU为增长式占用
    gpus = tf.config.experimental.list_physical_devices("GPU")  # 获取GPU设备列表
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    return


class Seq2Seq(Model):
    def __init__(self, n_input, n_output, n_units):
        super(Seq2Seq, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_units = n_units

        # Encoder网络层
        self.e_lstm = layers.LSTM(n_units, return_state=True)  # return_state设为True时才会返回最后时刻的状态h,c

        # Decoder网络层
        self.d_lstm = layers.LSTM(n_units, return_sequences=True, return_state=True)
        self.d_dense = layers.Dense(self.n_output, activation="softmax")

    def encoder(self, inputs):  # 编码器
        _, state_h, state_c = self.e_lstm(inputs)
        return state_h, state_c

    def decoder(self, inputs):  # 解码器
        input, input_state_h, input_state_c = inputs
        input_state = [input_state_h, input_state_c]
        h, output_state_h, output_state_c = self.d_lstm(input, initial_state=input_state)
        decoder_out = self.d_dense(h)
        return decoder_out, output_state_h, output_state_c

    def call(self, inputs, training=None, mask=None):  # seq2seq模型调用逻辑
        encoder_input, decoder_input = inputs
        state_h, state_c = self.encoder(encoder_input)
        decoder_output, _, _ = self.decoder([decoder_input, state_h, state_c])
        return decoder_output


class Translate:
    def __init__(self, data_path, n_units, num_samples=None):
        self.data_path = data_path  # 训练数据路径
        self.n_units = n_units  # seq2seq模型LSTM隐层节点数
        self.num_samples = num_samples  # 如果num_samples=None，则读取全部数据
        self.load_data()  # 读取数据
        self.build_model()  # 向量化

    def load_data(self):  # 读取数据
        df = pd.read_table(self.data_path, header=None)  # 读取txt数据
        if not self.num_samples: self.num_samples = len(df)  # 如果num_samples=None，则读取全部数据
        df = df.iloc[:self.num_samples, :, ]
        df.columns = ["inputs", "targets"]  # 添加列名
        df["targets"] = df["targets"].apply(lambda x: "\t" + x + "\n")  # 给targets列每个句子添加起止标记

        self.input_texts = df.inputs.values.tolist()  # 转化成输入列表
        self.target_texts = df.targets.values.tolist()  # 转化成目标列表

        self.input_characters = sorted(list(set(df.inputs.unique().sum())))  # 输入字符集
        self.target_characters = sorted(list(set(df.targets.unique().sum())))  # 目标字符集

        self.input_length = max(len(seq) for seq in self.input_texts)  # 输入语句最大长度
        self.output_length = max(len(seq) for seq in self.target_texts)  # 目标语句最大长度

        self.input_feature_length = len(self.input_characters)  # 输入字符集长度
        self.output_feature_length = len(self.target_characters)  # 目标字符集长度

        self.input_char2index = {char: index for index, char in enumerate(self.input_characters)}  # 输入字符-索引字典
        self.input_index2char = {index: char for index, char in enumerate(self.input_characters)}  # 输入索引-字符字典
        self.target_char2index = {char: index for index, char in enumerate(self.target_characters)}  # 目标字符-索引字典
        self.target_index2char = {index: char for index, char in enumerate(self.target_characters)}  # 目标字符-索引字典

    def vectorize(self):  # 向量化
        self.encoder_input = np.zeros([self.num_samples, self.input_length, self.input_feature_length])  # 向量化的encoder输入
        self.decoder_input = np.zeros([self.num_samples, self.output_length, self.output_feature_length])  # 向量化的decoder输入
        self.decoder_output = np.zeros([self.num_samples, self.output_length, self.output_feature_length])  # 向量化的decoder输出
        # 注意：训练模式下，以真实的上一个target作为decoder输入，进行强制学习，因此需要decoder_input和decoder_output
        # 而推理模式下，以上一个decoder的输出作为decoder输入，进行循环推理

        for seq_index, seq in enumerate(self.input_texts):  # 输入独热编码
            for char_index, char in enumerate(seq):
                self.encoder_input[seq_index, char_index, self.input_char2index[char]] = 1.

        for seq_index, seq in enumerate(self.target_texts):  # 目标独热编码
            for char_index, char in enumerate(seq):
                self.decoder_input[seq_index, char_index, self.target_char2index[char]] = 1.  # decoder_input包含起始"\t"字符
                if char_index > 0:
                    # decoder_output不包含起始"\t"字符，整体前移一位时序，作为ground truth监督
                    self.decoder_output[seq_index, char_index - 1, self.target_char2index[char]] = 1.

    def build_model(self):  # 创建模型
        self.model = Seq2Seq(n_input=self.input_feature_length, n_output=self.output_feature_length, n_units=self.n_units)

    def train(self, batch_size=8, epochs=200, validation_split=0.2):  # 训练
        self.vectorize()  # 向量化

        self.model.compile(optimizer="rmsprop",
                           loss="categorical_crossentropy")

        self.model.fit(x=[self.encoder_input, self.decoder_input],
                       y=self.decoder_output,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_split=validation_split)

        self.save_weights()  # 保存模型

    def infer(self, source):  # 翻译
        # 将source向量化（独热编码）
        encoder_input = np.zeros([1, self.input_length, self.input_feature_length])
        for char_index, char in enumerate(source):
            encoder_input[0, char_index, self.input_char2index[char]] = 1.

        state_h, state_c = self.model.encoder(encoder_input)  # 根据encoder，获得整个输入source语句的编码结果，即最后输出状态
        # 起始时：decoder输入为："\t"字符和encoder推理的编码，之后decoder的输入为：上次输出字符和上次状态
        decoder_input = np.zeros([1, 1, self.output_feature_length])
        decoder_input[0, 0, self.target_char2index["\t"]] = 1

        output = ""  # 翻译结果字符串
        for i in range(self.output_length):
            decoder_output, state_h, state_c = self.model.decoder([decoder_input, state_h, state_c])
            char_index = np.argmax(decoder_output[0, 0, :])  # 获得softmax最大索引
            char = self.target_index2char[char_index]  # 推理的字符
            output += char  # 拼接到output
            # 将当前推理的字符作为下次输入的字符，一直循环
            decoder_input = np.zeros([1, 1, self.output_feature_length])
            decoder_input[0, 0, char_index] = 1
            if char == "\n":  # 遇到终止符"\n"则停下来
                break
        return output

    def save_weights(self):  # 保存模型
        if not os.path.exists("model"):
            os.mkdir("model")
        self.model.save_weights("model/seq2seq.h5")

    def load_weights(self):  # 加载模型
        # 需使用实例数据恢复网络图
        dummy_encoder_input = np.zeros([1, self.input_length, self.input_feature_length])
        dummy_decoder_input = np.zeros([1, self.output_length, self.output_feature_length])
        self.model([dummy_encoder_input, dummy_decoder_input])
        self.model.load_weights("model/seq2seq.h5")

    def main(self):
        def isValid(source, vocab):  # 检查输入合法性
            if source == "":  # 只输入回车键也被判定不合法，python的input()默认把"\n"删除，所以需要用""判断
                return False
            for char in source:  # 遍历检查每个字符是否在vacab中
                if char not in vocab:
                    return False
            return True

        self.load_weights()  # 加载模型
        print("--------英译中v1.0，作者：RanYuan--------")
        print("（请输入英文句子，点击回车键即得中文翻译结果，输入exit则退出）")

        while True:
            while True:  # 直至输入合法才跳出循环
                source = input()
                if isValid(source, vocab=self.input_characters):
                    break
                else:
                    print("--------请输入合法英文句子，不要含有中文字符！--------")

            if source == "exit":  # 输入的是exit则结束
                print("--------感谢使用！--------")
                break
            target = self.infer(source)
            print(target)  # 输入翻译结果


if __name__ == "__main__":
    allow_memory_growth()
    # 如果内存溢出，适当减少训练数据规模num_samples、隐层神经元个数n_units、批量大小batch_size
    # 推荐配置：n_units=256, num_samples=None, batch_size=64, epochs=200, validation_split=0.2
    tran = Translate(data_path="data/corpus.txt", n_units=256, num_samples=1000)
    # tran.train(batch_size=8, epochs=200, validation_split=0.2)  # 训练模型，在使用模型时注释掉该行代码
    tran.main()  # 使用模型