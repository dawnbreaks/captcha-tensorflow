# -*- coding:utf-8 -*-
from gen_captcha import gen_captcha_text_and_image
from gen_captcha import number
from gen_captcha import alphabet
from gen_captcha import ALPHABET

import tensorflow as tf
import numpy as np
import math

# 图像大小
IMAGE_WIDTH = 80
IMAGE_HEIGHT = 80
FONT_SIZE = 76
CAPTCHA_TEXT_LEN = 1
CHAR_SET = number   # + alphabet + ALPHABET + ['_']  # 如果验证码长度小于4, '_'用来补齐
CHAR_SET_LEN = len(CHAR_SET)



# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


"""
cnn在图像大小是2的倍数时性能最高, 如果你用的图像大小不是2的倍数，可以在图像边缘补无用像素。
np.pad(image,((2,3),(2,2)), 'constant', constant_values=(255,))  # 在图像上补2行，下补3行，左补2行，右补2行
"""




def text2vec(text):
    text_len = len(text)
    if text_len > CAPTCHA_TEXT_LEN:
        raise ValueError('captcha text is too long.')

    vector = np.zeros(CAPTCHA_TEXT_LEN * CHAR_SET_LEN)

    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + CHAR_SET.index(c)
        vector[idx] = 1
    return vector


# 向量转回文本
def vec2text(vec):
    vec = np.reshape(vec, (CAPTCHA_TEXT_LEN, CHAR_SET_LEN))
    text = []
    for i in np.argmax(vec, 1):
        text.append(CHAR_SET[i])
    # char_pos = vec.nonzero()[0]
    # for i, c in enumerate(char_pos):
    #     char_at_pos = i  # c/63
    #     char_idx = c % CHAR_SET_LEN
    #     if char_idx < 10:
    #         char_code = char_idx + ord('0')
    #     elif char_idx < 36:
    #         char_code = char_idx - 10 + ord('A')
    #     elif char_idx < 62:
    #         char_code = char_idx - 36 + ord('a')
    #     elif char_idx == 62:
    #         char_code = ord('_')
    #     else:
    #         raise ValueError('error')
    #     text.append(chr(char_code))
    return "".join(text)


"""
#向量（大小MAX_CAPTCHA*CHAR_SET_LEN）用0,1编码 每63个编码一个字符，这样顺利有，字符也有
vec = text2vec("F5Sd")
text = vec2text(vec)
print(text)  # F5Sd
vec = text2vec("SFd5")
text = vec2text(vec)
print(text)  # SFd5
"""


# 生成一个训练batch
def get_next_batch(batch_size=100):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, CAPTCHA_TEXT_LEN * CHAR_SET_LEN])

    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image(IMAGE_WIDTH, IMAGE_HEIGHT, CHAR_SET, CAPTCHA_TEXT_LEN, FONT_SIZE)
            if image.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)

        batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y


####################################################################

X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, CAPTCHA_TEXT_LEN * CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)  # dropout


# 定义CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    # w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH)) #
    # w_c2_alpha = np.sqrt(2.0/(3*3*32))
    # w_c3_alpha = np.sqrt(2.0/(3*3*64))
    # w_d1_alpha = np.sqrt(2.0/(8*32*64))
    # out_alpha = np.sqrt(2.0/1024)

    # 3 conv layer
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    h_conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    h_conv1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    h_conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(h_conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    h_conv2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Fully connected layer
    fully_connected_layer_size = int(math.ceil(IMAGE_WIDTH / 4) * math.ceil(IMAGE_HEIGHT / 4) * 64)
    w_d = tf.Variable(w_alpha * tf.random_normal([fully_connected_layer_size, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(h_conv2, [-1, fully_connected_layer_size])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))

    #dropout layer
    dense = tf.nn.dropout(dense, keep_prob)

    #Readout Layer
    w_out = tf.Variable(w_alpha * tf.random_normal([1024, CAPTCHA_TEXT_LEN * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([CAPTCHA_TEXT_LEN * CHAR_SET_LEN]))
    y_conv = tf.add(tf.matmul(dense, w_out), b_out)
    # out = tf.nn.softmax(out)
    return y_conv


# 训练
def train_crack_captcha_cnn():
    print 'image shape(%d, %d), captchaTextLen: %d, charSetLen: %d, charset: %s' %(IMAGE_HEIGHT, IMAGE_WIDTH, CAPTCHA_TEXT_LEN, CHAR_SET_LEN, CHAR_SET)
    print 'inputSize: %d, outputSize: %d' %(IMAGE_HEIGHT * IMAGE_WIDTH, CAPTCHA_TEXT_LEN * CHAR_SET_LEN)

    output = crack_captcha_cnn()
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))
    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    # 最后一层用来分类的softmax和sigmoid有什么不同？
    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    predict = tf.reshape(output, [-1, CAPTCHA_TEXT_LEN, CHAR_SET_LEN])
    max_idx_predict = tf.argmax(predict, 2)
    max_idx_expected = tf.argmax(tf.reshape(Y, [-1, CAPTCHA_TEXT_LEN, CHAR_SET_LEN]), 2)
    correct_predict = tf.equal(max_idx_predict, max_idx_expected)
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        while True:
            batch_x, batch_y = get_next_batch(100)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5})
            print 'loss(%d, %f)' % (step, loss_)
            # 每100 step计算一次准确率
            if step % 10 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print 'accuracy(%d, %f)' %(step, acc)
                # 如果准确率大于50%,保存模型,完成训练
                if acc > 0.98:
                    saver.save(sess, "crack_capcha.model", global_step=step)
                    break

            step += 1


def crack_captcha(session, predict, captcha_image):
    text_list = session.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})
    text = text_list[0].tolist()
    vector = np.zeros(CAPTCHA_TEXT_LEN * CHAR_SET_LEN)
    i = 0
    for n in text:
        vector[i * CHAR_SET_LEN + n] = 1
        i += 1
    return vec2text(vector)

def restore_latest_checkpoint():
    output = crack_captcha_cnn()
    saver = tf.train.Saver()
    session = tf.Session()
    saver.restore(session, tf.train.latest_checkpoint('.'))
    predict = tf.argmax(tf.reshape(output, [-1, CAPTCHA_TEXT_LEN, CHAR_SET_LEN]), 2)
    return session, predict


def crack_captcha_test():
    session, predict = restore_latest_checkpoint()
    failed_count = 0
    for i in range(100):
        text, image = gen_captcha_text_and_image(IMAGE_WIDTH, IMAGE_HEIGHT, CHAR_SET, CAPTCHA_TEXT_LEN, FONT_SIZE)
        image = convert2gray(image)
        image = image.flatten() / 255
        predict_text = crack_captcha(session, predict, image)
        if text != predict_text:
            failed_count+=1
        print("expected: {}  predicted: {}  failed_count:{}".format(text, predict_text, failed_count))


if __name__ == '__main__':
    # print '%s' % vec2text(text2vec("0"))
    crack_captcha_test()
    # train_crack_captcha_cnn()




