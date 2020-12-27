LEN_THRESHOLD = 10


def read_files(data_dir):
    with open(data_dir + 'pos_pos', 'r') as f:
        pos_pos = f.readlines()
        pos_pos = [list(map(int, sen.split(' '))) for sen in pos_pos]
    with open(data_dir + 'pos_neg', 'r') as f:
        pos_neg = f.readlines()
        pos_neg = [list(map(int, sen.split(' '))) for sen in pos_neg]
    with open(data_dir + 'neg_pos', 'r') as f:
        neg_pos = f.readlines()
        neg_pos = [list(map(int, sen.split(' '))) for sen in neg_pos]
    with open(data_dir + 'neg_neg', 'r') as f:
        neg_neg = f.readlines()
        neg_neg = [list(map(int, sen.split(' '))) for sen in neg_neg]
    return pos_pos, pos_neg, neg_pos, neg_neg


# positive: 1, negative: 0
# afro-american: 1, white: 0
def get_labeled_data(pos_pos, pos_neg, neg_pos, neg_neg, total, train_s, task):
    data_train = []
    label_train = []
    data_test = []
    label_test = []

    if task == 'mention':
        for data in pos_pos[:train_s]:
            data_train.append(data)
            label_train.append(1)
        for data in pos_pos[train_s:total]:
            data_test.append(data)
            label_test.append(1)
        for data in pos_neg[:train_s]:
            data_train.append(data)
            label_train.append(1)
        for data in pos_neg[train_s:total]:
            data_test.append(data)
            label_test.append(1)

        for data in neg_pos[:train_s]:
            data_train.append(data)
            label_train.append(0)
        for data in neg_pos[train_s:total]:
            data_test.append(data)
            label_test.append(0)
        for data in neg_neg[:train_s]:
            data_train.append(data)
            label_train.append(0)
        for data in neg_neg[train_s:total]:
            data_test.append(data)
            label_test.append(0)

        return data_train, label_train, data_test, label_test

    elif task == 'race':
        for data in pos_pos[:train_s]:
            data_train.append(data)
            label_train.append(1)
        for data in pos_pos[train_s:total]:
            data_test.append(data)
            label_test.append(1)
        for data in neg_pos[:train_s]:
            data_train.append(data)
            label_train.append(1)
        for data in neg_pos[train_s:total]:
            data_test.append(data)
            label_test.append(1)

        for data in pos_neg[:train_s]:
            data_train.append(data)
            label_train.append(0)
        for data in pos_neg[train_s:total]:
            data_test.append(data)
            label_test.append(0)
        for data in neg_neg[:train_s]:
            data_train.append(data)
            label_train.append(0)
        for data in neg_neg[train_s:total]:
            data_test.append(data)
            label_test.append(0)

        return data_train, label_train, data_test, label_test

    else:
        label2_train = []
        label2_test = []
        for data in pos_pos[:train_s]:
            data_train.append(data)
            label_train.append(1)
            label2_train.append(1)
        for data in pos_pos[train_s:total]:
            data_test.append(data)
            label_test.append(1)
            label2_test.append(1)
        for data in pos_neg[:train_s]:
            data_train.append(data)
            label_train.append(1)
            label2_train.append(0)
        for data in pos_neg[train_s:total]:
            data_test.append(data)
            label_test.append(1)
            label2_test.append(0)

        for data in neg_pos[:train_s]:
            data_train.append(data)
            label_train.append(0)
            label2_train.append(1)
        for data in neg_pos[train_s:total]:
            data_test.append(data)
            label_test.append(0)
            label2_test.append(1)
        for data in neg_neg[:train_s]:
            data_train.append(data)
            label_train.append(0)
            label2_train.append(0)
        for data in neg_neg[train_s:total]:
            data_test.append(data)
            label_test.append(0)
            label2_test.append(0)

        return data_train, label_train, label2_train, data_test, label_test, label2_test


def get_data(task, data_dir):
    """
    routing the task and dir to the corresponding function.
    returning train and test data
    """
    train_s = 41500
    total = 44000

    pos_pos, pos_neg, neg_pos, neg_neg = read_files(data_dir)

    return get_labeled_data(pos_pos, pos_neg, neg_pos, neg_neg, total, train_s, task)
