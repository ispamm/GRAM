import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from data.loader import  MetaLoader, PrefetchLoader
from data import data_registry
from utils.distributed import DistributedSampler_wopadding
from .logger import LOGGER


    
def create_train_dataloaders(args):
    data_cfg = args.data_cfg.train
    dataloaders = []
    dataloaders_dict={}
    train_steps = []
    loader_names = []
    
    if len(data_cfg) == 0:
        return 

    for d_cfg in data_cfg:
      
        dataset_ls = []

        name = d_cfg['name']
        dataset = data_registry[d_cfg.type](d_cfg, args)

        # print(dataset[0])

        collate_fn = dataset.collate_fn
        worker_init_fn = dataset.worker_init_fn
        use_sampler = dataset.use_sampler


        LOGGER.info("Create Dataset {} Success".format(name))
        task = d_cfg['task']
        batch_size = d_cfg['batch_size']
        n_workers = d_cfg['n_workers'] 

        if 'steps' in d_cfg:
            train_steps.append(d_cfg['steps'])
        elif 'epoch' in d_cfg:
            epoch = d_cfg['epoch']
            train_steps.append(int((len(dataset) // batch_size) * epoch))

        loader = build_dataloader(dataset, collate_fn, True, batch_size // args.run_cfg.gradient_accumulation_steps , n_workers, worker_init_fn, use_sampler)

        dataloaders.append(loader)
        loader_names.append(f'{task}--{name}')
    
    
    for i in range(len(dataloaders)):
        ratio = train_steps[i]
        dataloaders_dict[loader_names[i]] = (dataloaders[i], ratio)

    n_gpu = dist.get_world_size()
    for name, (loader, ratio) in dataloaders_dict.items():
        # epoch = (ratio * loader.batch_size * n_gpu ) // len(loader.dataset)
        LOGGER.info(f" loader {name} , ratio {ratio} , bs_pergpu {loader.batch_size}, n_workers {loader.num_workers}" )


    meta_loader = MetaLoader(dataloaders_dict,
                            accum_steps=args.run_cfg.gradient_accumulation_steps,
                            distributed=n_gpu > 1)
    
    if args.run_cfg.num_train_steps == 0:
        total_train_steps = sum(train_steps)
        args.run_cfg.num_train_steps = total_train_steps

    
        
    meta_loader = PrefetchLoader(meta_loader)
    meta_loader.ndata = len(dataloaders_dict)
    args.run_cfg.valid_steps = args.run_cfg.num_train_steps // args.run_cfg.valid_freq -1
    
 
    
    return meta_loader


def create_val_dataloaders(args):
    data_cfg = args.data_cfg.val
    dataloaders = {}
    for d_cfg in data_cfg:
        name = d_cfg['name']
        dataset = data_registry[d_cfg.type](d_cfg, args)
        collate_fn = dataset.collate_fn
        worker_init_fn = dataset.worker_init_fn
        use_sampler = dataset.use_sampler
        # task = d_cfg['task'].split('_') 
        # if 'qa' in task:
        #     dataset.make_submission = d_cfg.get('make_submission', False)

        # if 'cap' in task:
        #     dataset.annfile = d_cfg['annfile']

        # dataset.data_type = data_type
        dataset.name = name
        LOGGER.info("Create Dataset {} Success. Len {}".format(name, len(dataset)))
        task = d_cfg['task']
        batch_size = d_cfg['batch_size']
        n_workers = d_cfg['n_workers'] 
        loader = build_dataloader(dataset, collate_fn, False, batch_size, n_workers, worker_init_fn, use_sampler)
        task_name = f'{task}--{name}'
        dataloaders[task_name] = PrefetchLoader(loader)
    return dataloaders


def build_dataloader(dataset, collate_fn, is_train, batch_size, n_workers=None, worker_init_fn=None, use_sampler=True):
    batch_size = batch_size // dist.get_world_size()
    if use_sampler:
        if is_train:
            sampler = DistributedSampler(dataset,shuffle=True)
            #sampler.set_epoch(epoch)
        else:
            sampler = DistributedSampler_wopadding(dataset)
        if is_train:
            loader = DataLoader(dataset, sampler = sampler, batch_size = batch_size,
                            num_workers=n_workers, pin_memory=True,
                            collate_fn=collate_fn, drop_last=is_train,worker_init_fn=worker_init_fn)#,shuffle=True)
        else:
            loader = DataLoader(dataset, sampler = sampler, batch_size = batch_size,
                            num_workers=n_workers, pin_memory=True,
                            collate_fn=collate_fn, drop_last=is_train,worker_init_fn=worker_init_fn)

    else:
        if is_train:
            loader = DataLoader(dataset,  batch_size = batch_size,
                            num_workers=n_workers, pin_memory=True,
                            collate_fn=collate_fn, drop_last=is_train,worker_init_fn=worker_init_fn)#,shuffle=True)  
        else:
            loader = DataLoader(dataset, sampler = sampler, batch_size = batch_size,
                            num_workers=n_workers, pin_memory=True,
                            collate_fn=collate_fn, drop_last=is_train,worker_init_fn=worker_init_fn)  

    return loader

