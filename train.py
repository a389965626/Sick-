# -*- coding: utf-8 -*-
# @Time    : 2020/1/20 19:10
# @Author  : chenjunyu
# @FileName: train
# @Software: PyCharm


from utilities import *
from MSEM_WI_model import *
import os
import time
from configs import config
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import datetime
import numpy as np


# 预处理
def preprocess():
    # Load data
    print("Loading data...")
    '''
    s1, s2, y, lables = read_msrp('./msr_paraphrase_train.txt')    # 读取数据
    s1_test, s2_test, y_test, lables_test= read_msrp('./msr_paraphrase_test.txt', is_Train=False)
    '''
    s1, s2, y, lables= read_sick('./SICK_train.txt')  # 读取数据
    s1_test, s2_test, y_test, lables_test = read_sick('./SICK_test_annotated_2.txt', is_Train=False)

    # Split train/dev set
    # 最好是随机选择一部分用于dev验证（这里先不管验证，先跑起来再说）
    '''
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    # 前半部分是训练集，后半部分是测试集
    s1_train, s1_dev = s1[:dev_sample_index], s1[dev_sample_index:]
    s2_train, s2_dev = s2[:dev_sample_index], s2[dev_sample_index:]
    y_train, y_dev = y[:dev_sample_index], y[dev_sample_index:]
    '''
    return s1, s2, y, lables, s1_test, s2_test, y_test, lables_test


def train(s1, s2, y, lables, s1_test, s2_test, y_test, lables_test, words1, words2, color):

    print("Training...")
    starttime = datetime.datetime.now()
    # Training
    # ==================================================
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=config.allow_soft_placement,
            log_device_placement=config.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            my_model = MSEM_WI(config, word_embedding)

            # Define Training procedure（定义训练步骤，实际就是定义训练的optimizer）
            # 以下等价于 train_op = optimizer.minimize(loss, global_step=global_step)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(config.learning_rate)  # 优化算法
            grads_and_vars = optimizer.compute_gradients(my_model.loss)  # 梯度，方差
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Output directory for models and summaries（将模型和总结写到该目录）
            timestamp = str(int(time.time()))  # 记录当前时间的时间戳
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy（保存loss和accuracytf.summary.scalar用来显示标量信息）
            loss_summary = tf.summary.scalar("loss", my_model.loss)
            mse_summary = tf.summary.scalar("accuracy", my_model.mse)
            # f1_summary = tf.summary.scalar("f1", my_model.F1)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, mse_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, mse_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=config.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())


            def train_step(s1_batch, s2_batch, y_batch, lables_batch):
                """
                A single training step
                """
                feed_dict = {
                    my_model.sentence_one_word: s1_batch,
                    my_model.sentence_two_word: s2_batch,
                    my_model.y_true: lables_batch,
                    my_model.y: y_batch,
                    my_model.is_train: True
                }
                # global_step用于记录全局的step数，就是当前运行到的step
                _, step, summaries, loss, mse = sess.run(
                    [train_op, global_step, train_summary_op, my_model.loss, my_model.mse],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, mse {:g}".format(time_str, step, loss, mse))
                train_summary_writer.add_summary(summaries, step)
                endtime = datetime.datetime.now()
                print('训练时间: ', (endtime - starttime).seconds/60)

            def test_step(s1_batch, s2_batch, y_batch, lables_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    my_model.sentence_one_word: s1_batch,
                    my_model.sentence_two_word: s2_batch,
                    my_model.y_true: lables_batch,
                    my_model.y: y_batch,
                    my_model.is_train: False
                }
                step, summaries, loss, mse, se, y, final_output, heatmap1, heatmap2, heatmap1_total, heatmap2_total = sess.run(
                    [global_step, dev_summary_op, my_model.loss, my_model.mse, my_model.se, my_model.y, my_model.final_output_reshape, my_model.heat_matrix1, my_model.heat_matrix2, my_model.heatmap1_total, my_model.heatmap2_total],
                    feed_dict)
                # print(y)
                # print(final_output)
                time_str = datetime.datetime.now().isoformat()
                # print("loss and acc in test_batch:")
                # print("{}: step {}, loss {:g}, mse {:g}, se {:g}".format(time_str, step, loss, mse, se))
                if writer:
                    writer.add_summary(summaries, step)
                return se, y, final_output, heatmap1, heatmap2, heatmap1_total, heatmap2_total   # y, final_output 计算整体的均方误差

            # Generate batches
            batches = batch_iter(s1, s2, y, lables, config.batch_size, config.num_epochs)
            # Training loop. For each batch...
            for s1_batch, s2_batch, y_batch, lables_batch in batches:
                train_step(s1_batch, s2_batch, y_batch, lables_batch)
                current_step = tf.train.global_step(sess, global_step)    # 一个step就是一个batch

                # 每隔100个step，就对测试集进行一次测试。测试集也需要分batch读入，否则会内存不够
                # 测试集分批读入
                if current_step > 500:
                    if current_step % config.evaluate_every == 0:
                        print("\nTesting:")
                        '''
                        train_batches = batch_iter(s1, s2, y, lables, config.batch_size, 1, shuffle=True)
                        se_total_train = 0
                        batch_num_train = 0
                        for s1_batch, s2_batch, y_batch, lables_batch in train_batches:
                            se_train = test_step(s1_batch, s2_batch, y_batch, lables_batch)
                            batch_num_train = batch_num_train + 1
                            se_total_train = se_total_train + se_train
                        print('batch_train:', batch_num_train)
                        print("total_train:", batch_num_train * config.batch_size)
                        print('se_total_train:', se_total_train)
                        train_mse = se_total_train / (batch_num_train * config.batch_size)
                        print('train mse:', train_mse)
                        '''
                        test_batches = batch_iter(s1_test, s2_test, y_test, lables_test, config.batch_size, 1, shuffle=False)
                        # correct_total_num = 0
                        se_total = 0
                        batch_num = 0
                        y_total = []
                        y_total = np.array(y_total)
                        final_output_total = []
                        final_output_total = np.array(final_output_total)
                        # yhat_total = []
                        # yhat_total = np.array(yhat_total)

                        print('testing: ')
                        for s1_batch, s2_batch, y_batch, lables_batch in test_batches:
                            se, y, final_output, heatmap1, heatmap2, heatmap1_total, heatmap2_total = test_step(s1_batch, s2_batch, y_batch, lables_batch)
                            y_total = np.append(y_total, y)
                            final_output_total = np.append(final_output_total, final_output)
                            # yhat_total = np.append(yhat_total, yhat)
                            # print('y_total', y_total)
                            # print('len_y_totoal:', len(y_total))
                            # print(final_output_total)
                            # print('final_output_totoal:', final_output_total)
                            batch_num = batch_num + 1   # batch数自增
                            se_total = se_total + se
                        print('y_total', y_total)
                        print('final_output_totoal:', final_output_total)
                        print('heatmap1:', heatmap1[0])
                        print('heatmap1_total', heatmap1_total[0])
                        print('heatmap2:', heatmap2[0])
                        print('heatmap2_total', heatmap2_total[0])
                        # 只把符合要求的进行可视化
                        if(final_output_total[0] > 1.1 and final_output_total[0] < 1.3):
                            # 依次遍历s1和s2的每一个语义
                            for i in range(config.attention_num):   # 0-9
                                attention1 = heatmap1[0][i][:len(words1)]   # 我们测试句子的第i个语义
                                attention2 = heatmap2[0][i][:len(words2)]
                                path1 = 's1-' + str(final_output_total[0]) + '-' + str(i) + '.tex'
                                path2 = 's2-' + str(final_output_total[0]) + '-' + str(i) + '.tex'
                                generate(words1, attention1, path1, color, rescale_value=True)
                                generate(words2, attention2, path2, color, rescale_value=True)
                            path1_total = 's1-' + str(final_output_total[0]) + '-total' + '.tex'
                            path2_total = 's2-' + str(final_output_total[0]) + '-total' + '.tex'
                            attention1_total = heatmap1_total[0][0][:len(words1)]
                            attention2_total = heatmap2_total[0][0][:len(words2)]
                            generate(words1, attention1_total, path1_total, color, rescale_value=True)
                            generate(words2, attention2_total, path2_total, color, rescale_value=True)

                        print('batch:', batch_num)
                        print("total:", batch_num * config.batch_size)
                        print('se_total:', se_total)
                        test_mse = se_total / (batch_num * config.batch_size)
                        print('test mse:', test_mse)
                        print('test mse direct:', mse_calculate(y_total, final_output_total))
                        print('pearson:', pearsonr(y_total, final_output_total)[0])
                        print('spearman:', spearmanr(y_total, final_output_total)[0])
                    '''
                    if current_step % config.checkpoint_every == 0:  # 每隔100个step，就保存模型
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))
                    '''


def mse_calculate(y, predict_y):
    return ((predict_y - y) ** 2).mean()


def generate(text_list, attention_list, latex_file, color='red', rescale_value=False):
    assert(len(text_list) == len(attention_list))
    if rescale_value:
        attention_list = rescale(attention_list)
    word_num = len(text_list)
    text_list = clean_word(text_list)
    with open(latex_file, 'w') as f:
        f.write(r'''\documentclass[varwidth]{standalone}
\special{papersize=210mm,297mm}
\usepackage{color}
\usepackage{tcolorbox}
\usepackage{CJK}
\usepackage{adjustbox}
\tcbset{width=0.9\textwidth,boxrule=0pt,colback=red,arc=0pt,auto outer arc,left=0pt,right=0pt,boxsep=5pt}
\begin{document}
\begin{CJK*}{UTF8}{gbsn}'''+'\n')
        string = r'''{\setlength{\fboxsep}{0pt}\colorbox{white!0}{\parbox{0.9\textwidth}{'''+"\n"
        for idx in range(word_num):
            string += "\\colorbox{%s!%s}{"%(color, attention_list[idx])+"\\strut " + text_list[idx]+"} "
        string += "\n}}}"
        f.write(string+'\n')
        f.write(r'''\end{CJK*}
\end{document}''')

def rescale(input_list):
    the_array = np.asarray(input_list)
    the_max = np.max(the_array)
    the_min = np.min(the_array)
    rescale = (the_array - the_min)/(the_max-the_min)*100
    return rescale.tolist()


def clean_word(word_list):
    new_word_list = []
    for word in word_list:
        for latex_sensitive in ["\\", "%", "&", "^", "#", "_",  "{", "}"]:
            if latex_sensitive in word:
                word = word.replace(latex_sensitive, '\\'+latex_sensitive)
        new_word_list.append(word)
    return new_word_list

if __name__ == '__main__':
    latex_special_token = ["!@#$%^&*()"]
    s1, s2, y, lables, s1_test, s2_test, y_test, lables_test = preprocess()  # 预处理

    sent1 = 'The girl who is little is combing her hair into a pony tail'
    sent2 = 'A man in a red shirt is doing a trick with the rollerblades'

    words1 = sent1.split()
    print(words1)
    word_num1 = len(words1)

    words2 = sent2.split()
    print(words2)
    word_num2 = len(words2)

    color = 'red'

    train(s1, s2, y, lables, s1_test, s2_test, y_test, lables_test, words1, words2, color)
    # generate(words1, attention1, "s1_6.tex", color, rescale_value=True)


