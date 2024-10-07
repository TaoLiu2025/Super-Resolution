from importlib import import_module

#from dataloader import MSDataLoader
#from torch.utils.data.dataloader import default_collate
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset
import utility
# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            #datasets = []
            module_train = import_module('data.' + args.data_train.lower())   ## load the right dataset loader module
            trainset = getattr(module_train, args.data_train)(args)   ### load the dataset, args.data_train is the  dataset name
            #datasets.append(getattr(module_train, args.data_train)(args))
            #import pdb;pdb.set_trace()
            '''
            self.loader_train = MSDataLoader(
                args,
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )
            '''
            
            self.loader_train = dataloader.DataLoader(
                #MyConcatDataset(datasets),
                trainset,
                batch_size=args.batch_size,               
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )

            

        if args.data_test in ['Set5', 'Set14', 'B100', 'Manga109', 'Urban100','wzry']:
            module_test = import_module('data.benchmark')
            testset = getattr(module_test, 'Benchmark')(args, name=args.data_test,train=False)
        else:
            module_test = import_module('data.' +  args.data_test.lower())
            testset = getattr(module_test, args.data_test)(args, train=False)

        '''
        self.loader_test = MSDataLoader(
            args,
            testset,
            batch_size=1,
            shuffle=False,
            pin_memory=not args.cpu
        )
        '''
        self.loader_test = []
        self.loader_test.append(
            dataloader.DataLoader(
                testset,
                batch_size = 1,
                shuffle=False,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )
        )

