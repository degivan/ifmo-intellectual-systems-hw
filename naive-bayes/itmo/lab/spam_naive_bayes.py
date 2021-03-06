import os

from math import log

import matplotlib.pyplot as plt


def train(train_data):
    total = 0
    mail_spam = 0
    subj_words_count = [0, 0]
    text_words_count = [0, 0]
    subject_words = [{}, {}]
    text_words = [{}, {}]
    for email in train_data:
        positive = email.label == 'SPAM'
        mail_spam += 1 * positive
        for word in email.subject:
            subject_words[positive][word] = subject_words[positive].get(word, 0) + 1
            subj_words_count[positive] += 1
        for word in email.text:
            text_words[positive][word] = text_words[positive].get(word, 0) + 1
            text_words_count[positive] += 1
        total += 1
    p_spam = mail_spam / float(total)
    p_ham = 1 - p_spam
    return Classifier(p_spam, p_ham, subj_words_count, subject_words, text_words_count, text_words)


class Classifier(object):
    def __init__(self, p_spam, p_ham, subj_wc, subj_w, text_wc, text_w):
        self.p_spam = p_spam
        self.p_ham = p_ham
        self.subj_wc = subj_wc
        self.subj_w = subj_w
        self.text_wc = text_wc
        self.text_w = text_w
        self.total_subj = sum(subj_wc)
        self.total_text = sum(text_wc)

    def classify(self, email, subj_coeff=3.0, trust_coeff=0.0):
        norm = 1 + subj_coeff * len(email.subject) + len(email.text)
        is_spam = self.cond_email(email, True, subj_coeff)
        is_ham = self.cond_email(email, False, subj_coeff)
        if (is_spam / norm) + trust_coeff > is_ham / norm:
            return 'SPAM'
        else:
            return 'HAM'

    def cond_email(self, email, spam, subj_coeff):
        sum = 0.0
        if spam:
            sum -= log(self.p_spam)
        else:
            sum -= log(self.p_ham)
        for word in email.subject:
            sum -= subj_coeff * log(self.cond_subj_word(word, spam), 10 ** (-7))
        for word in email.text:
            sum -= log(self.cond_text_word(word, spam), 10 ** (-7))
        return sum

    def cond_subj_word(self, word, spam, alpha=10 ** (-19)):
        word_count = self.subj_w[spam].get(word, 0)
        if word_count == 0:
            word_count += alpha
        return word_count / float(self.subj_wc[spam] + alpha * self.total_subj)

    def cond_text_word(self, word, spam, alpha=10 ** (-19)):
        word_count = self.text_w[spam].get(word, 0)
        word_count += alpha
        return word_count / float(self.text_wc[spam] + alpha * self.total_text)


class Email(object):
    def __init__(self, label, subject, text):
        self.label = label
        self.subject = subject
        self.text = text


def create_email_obj(title, content):
    if 'Legit' in title:
        label = 'HAM'
    else:
        label = 'SPAM'
    subj, text = content.split('\n\n')
    subject = map(int, subj[9:].split())
    text = map(int, text.split())
    return Email(label, subject, text)


def test(classifier, test_data, trust_coeff=0.0):
    correct_counts = {'HAM': 0, 'SPAM': 0}
    incorrect_counts = {'HAM': 0, 'SPAM': 0}
    for email in test_data:
        result = classifier.classify(email, trust_coeff=trust_coeff)
        is_corr_answ = email.label == result
        if is_corr_answ:
            correct_counts[result] = correct_counts[result] + 1
        else:
            incorrect_counts[result] = incorrect_counts[result] + 1
    total = float(len(test_data))
    spe = correct_counts['HAM'] / float(correct_counts['HAM'] + incorrect_counts['SPAM'])
    precision = correct_counts['SPAM'] / float(correct_counts['SPAM'] + incorrect_counts['SPAM'])
    recall = correct_counts['SPAM'] / float(correct_counts['SPAM'] + incorrect_counts['HAM'])
    f1_score = 2 * precision * recall / (precision + recall)
    accuracy = (correct_counts['SPAM'] + correct_counts['HAM']) / total
    return 1 - precision, f1_score, accuracy, spe, recall


def print_results(dh, f1, acc):
    dh_s, f1_s, acc_s = ["{0:0.5f}".format(i) for i in dh, f1, acc]
    print "{} -- {} -- {}".format(dh_s, f1_s, acc_s)


if __name__ == '__main__':
    data = [[] for i in range(10)]
    for i in range(1, 11):
        path = '../../data/part' + str(i)
        for f in os.listdir(path):
            opened_f = open(os.path.join(path, f), 'r')
            content = opened_f.read()
            title = f.title()
            opened_f.close()
            data[i - 1].append(create_email_obj(title, content))
    print("percentage of dismissed ham -- f1-score -- accuracy")
    x_axis = []
    y_axis = []
    for coeff in [-0.1 + 0.005 * x for x in range(40)]:
        avg_dh, avg_f1, avg_acc = 0.0, 0.0, 0.0
        avg_spe = 0.0
        avg_sen = 0.0
        for i in range(10):
            train_data = []
            test_data = data[i]
            for j in range(10):
                if j != i:
                    train_data += data[j]
            classifier = train(train_data)
            dh, f1, acc, spe, sen = test(classifier, test_data, trust_coeff=coeff)
            avg_spe += 1 - spe
            avg_sen += sen
            avg_dh += dh
            avg_f1 += f1
            avg_acc += acc
        print("In average:")
        print_results(avg_dh / 10, avg_f1 / 10, avg_acc / 10)
        x_axis.append(avg_spe / 10)
        y_axis.append(avg_sen / 10)
    plt.plot(x_axis, y_axis, 'ro')
    plt.grid(True)
    plt.show()
