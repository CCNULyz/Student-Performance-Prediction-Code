from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    def __init__(self, root_path_X, root_path_Y, time_step, flag='TRAIN', size=None,
                 target='class', scale=True, timeenc="W"):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        # init
        assert flag in ['TRAIN', 'TEST', 'VAL']
        type_map = {'TRAIN': 0, 'TEST': 1, 'VAL': 2}
        self.set_type = type_map[flag]

        self.target = target
        self.scale = scale
        self.timeenc = timeenc

        self.root_path_X = root_path_X
        self.root_path_Y = root_path_Y
        self.time_step = time_step
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # df_raw = pd.read_csv(os.path.join(self.root_path,
        #                                   self.data_path))
        np_raw_X = np.load(self.root_path_X)
        np_raw_Y = np.load(self.root_path_Y)
        Br, Tr, Fr = np_raw_X.shape
        # self.max_seq_len = Tr
        self.max_seq_len = self.time_step
        """
        np_raw_X.columns:[batch_size, time_step, features]
        """
        # cols = list(df_raw.columns)
        # cols.remove(self.target)
        # cols.remove('date')
        # df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(np_raw_X) * 0.7)
        num_test = int(len(np_raw_X) * 0.2)
        num_vali = len(np_raw_X) - num_train - num_test
        border1s = [0, num_train, num_test+num_train]
        border2s = [num_train, num_test+num_train, -1]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.scale:
            train_data = np_raw_X[border1:border2]
            B, T, F = train_data.shape
            self.scaler.fit(train_data.reshape(B, -1))
            data = self.scaler.transform(np_raw_X.reshape(Br, -1)).reshape(Br, Tr, Fr)
        else:
            data = np_raw_X

        """
        嵌入时间维度
        """
        # 创建一个新的数组来存放扩展后的数据
        if self.timeenc == 'W':
            B, T, F = data.shape
            data_stamp = np.zeros((B, T, 1))
            data_stamp = data_stamp.reshape(-1, 1)
            # 假设我们已经知道每个时间步的日期
            # 这里我们简单使用每个样本的索引模拟每周的周数，实际应用中你需要根据具体日期计算
            week_of_year = np.array([i % T + 1 for i in range(B * T)])
            """Week of year encoded as value between [-0.5, 0.5]"""
            stamp_weeks = ((week_of_year - 1) / 52 - 0.5)
            data_stamp[:, 0] = stamp_weeks
        elif self.timeenc == 'D':
            B, T, F = data.shape
            data_stamp = np.zeros((B, T, 1))
            data_stamp = data_stamp.reshape(-1, 1)
            # 假设我们已经知道每个时间步的日期
            # 这里我们简单使用每个样本的索引模拟每周的周数，实际应用中你需要根据具体日期计算
            day_of_year = np.array([i % T + 1 for i in range(B * T)])
            """Date of year encoded as value between [-0.5, 0.5]"""
            stamp_days = ((day_of_year - 1) / 365 - 0.5)
            data_stamp[:, 0] = stamp_days
        elif self.timeenc == 'WD':
            B, T, F = data.shape
            data_stamp = np.zeros((B, T, 2))
            data_stamp = data_stamp.reshape(-1, 2)
            # 假设我们已经知道每个时间步的日期
            # 这里我们简单使用每个样本的索引模拟每周的周数，实际应用中你需要根据具体日期计算
            day_of_year = np.array([i % T + 1 for i in range(B * T)])
            day_of_week = ((day_of_year - 1) % 7)
            """Date of year encoded as value between [-0.5, 0.5]"""
            stamp_day1 = ((day_of_year - 1) / 365 - 0.5)
            stamp_day2 = (day_of_week / 6 - 0.5)
            data_stamp[:, 0] = stamp_day1
            data_stamp[:, 1] = stamp_day2
        else:
            B, T, F = data.shape
            data_stamp = np.zeros((B, T, 1))
        data_stamp = data_stamp.reshape(B, T)

        self.data_x = data[border1:border2, :self.time_step]
        self.data_y = np_raw_Y[border1:border2]
        self.data_stamp = data_stamp[:, :self.time_step]

    def __getitem__(self, index):
        s_begin = index
        seq_x = self.data_x[s_begin]
        y = self.data_y[s_begin]
        seq_x_mark = self.data_stamp[s_begin]

        return seq_x, y, seq_x_mark

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


if __name__ == '__main__':
    flag = "test"
    data_set = Dataset_Custom(
        root_path_X="../data/实验用数据/X20.npy",
        root_path_Y="../data/实验用数据/Y20.npy",
        flag="test",
        time_step=24
    )
    shuffle_flag = True
    drop_last = True
    batch_size = 32  # bsz for train and valid

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=2,
        drop_last=drop_last,
    )
    print(data_set)

    for i, (a, b, c) in enumerate(data_loader):
        print(a.shape)
        print(b.shape)
        print(c.shape)
