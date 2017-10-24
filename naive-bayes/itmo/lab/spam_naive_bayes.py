import os


def train(train_data):
    total = 0
    num_spam = 0
    for email in train_data:
        if email.label == 'spmsg':
            num_spam += 1
        total += 1
    p_spam = num_spam / total
    p_ham = 1 - p_spam
    return p_spam, p_ham


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
    subj_and_text = content.split('\n\n')
    subject = map(int, subj_and_text[0][9:].split())
    text = map(int, subj_and_text[1].split())
    return Email(label, subject, text)


if __name__ == '__main__':
    train_data = [[] for i in range(10)]
    for i in range(1, 11):
        path = '../../data/part' + str(i)
        for f in os.listdir(path):
            opened_f = open(os.path.join(path, f), 'r')
            content = opened_f.read()
            title = f.title()
            opened_f.close()
            train_data[i - 1].append(create_email_obj(title, content))
    
