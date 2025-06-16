from data_provider.data_loader_new import Dataset_Custom
from torch.utils.data import DataLoader

data_dict = {
    'Custom': Dataset_Custom,
}


def data_provider(args, flag, timestep):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid

    if args.task_name == 'classification':
        data_set = Data(
            root_path_X=args.root_path_X,
            root_path_Y=args.root_path_Y,
            flag=flag,
            time_step=timestep
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )
        return data_set, data_loader

    else:
        data_set = Data(
            root_path_X=args.root_path_X,
            root_path_Y=args.root_path_Y,
            flag=flag,
            time_step=timestep
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )
        return data_set, data_loader
