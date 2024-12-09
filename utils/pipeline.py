import os
import torch 
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm 
# from apex import amp
from collections import defaultdict
from torch.nn.utils import clip_grad_norm_
from evaluation import evaluation_registry
from .save import ModelSaver
from .tool import NoOp
from .logger import LOGGER, RunningMeter
from .sched import get_lr_sched
from torch.cuda.amp import autocast, GradScaler
import wandb


def train(model, optimizer, train_loader, val_loaders, args, start_step=0, verbose_time=False):
    run_cfg = args.run_cfg
    dataset_cfg = args.data_cfg.train[0]
    if dist.get_rank() == 0:
        pbar = tqdm(total=run_cfg.num_train_steps, initial=start_step)
        model_saver = ModelSaver(os.path.join(run_cfg.output_dir, 'ckpt'),remove_before_ckpt=run_cfg.remove_before_ckpt)
    else:
        pbar = NoOp()
        model_saver = NoOp()
        
    loss_moving_averagetors ={}
    metric_logger_dict = defaultdict(dict)
    global_step = start_step

    scaler = GradScaler()

    best_indicator = {}
    evaluate_fn = evaluation_registry[model.config.evaluation_type]

    for step, (name, batch) in enumerate(train_loader):
      
        ndata = train_loader.ndata
        task = name.split('--')[0]



        if run_cfg.fp16:
            with autocast():
                loss_dict = model(batch, task=task, compute_loss=True)
                loss = sum(list(loss_dict.values()))
                loss_dict['total_loss'] = loss
                loss_dict = {k:v.item() for k,v in loss_dict.items()}
                if dist.get_rank() == 0:
                    wandb.log(loss_dict)
                
        else:
            loss_dict = model(batch, task=task, compute_loss=True)
            loss = sum(list(loss_dict.values()))
            loss_dict['total_loss'] = loss
            loss_dict = {k:v.item() for k,v in loss_dict.items()}
            if dist.get_rank() == 0:
                wandb.log(loss_dict)
            





        


        if not name in loss_moving_averagetors:
                ### first time initialize 
            for k in loss_dict.keys():
                loss_moving_averagetors[f'loss_{name}/{k}'] = RunningMeter()
        ####accumulate loss

        for k,v in loss_dict.items():
            loss_moving_averagetors[f'loss_{name}/{k}'](v)
    

        global_step += 1
        # learning rate scheduling
        lr_ratio = get_lr_sched(global_step, run_cfg)

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['init_lr'] * lr_ratio
        
        if global_step % 50 == 0:    
            LOGGER.info({name : averagetor.val for name, averagetor in loss_moving_averagetors.items()})                                   
        
        # update model params


        if run_cfg.fp16:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if not run_cfg.use_ddp:
            works = []
            for p in model.parameters():
            # to speed it up, you can also organize grads to larger buckets to make allreduce more efficient
                if p.grad is not None:
                    works.append(dist.all_reduce(p.grad, async_op=True))
            for work in works:
                work.wait()


        # if run_cfg.grad_norm != -1:
        #     grad_norm = clip_grad_norm_(model.parameters(), run_cfg.grad_norm)

        if run_cfg.fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
            optimizer.zero_grad()
        pbar.update(1)


        
        if (global_step+1) % run_cfg.valid_steps == 0:
            eval_log = evaluate_fn(model, val_loaders, run_cfg, global_step)

            if dist.get_rank() == 0:
                for task_name, val_log in eval_log.items():
                    for eval_name, metric in val_log.items():
                        eval_name = task_name +'_' +eval_name 
                        metric_logger_dict[eval_name][str(global_step)] = metric
                        LOGGER.info(f"====-evaluation--{eval_name}=====step {global_step}--===========\n")
                        LOGGER.info(metric)
                        best_name = get_best_name(eval_name, metric)
                        if best_name is not None:
                            if ('best_step' not in metric_logger_dict[eval_name]) or \
                                    (metric[best_name] >= metric_logger_dict[eval_name]['best_value']):
                                metric_logger_dict[eval_name]['best_step'] = global_step
                                metric_logger_dict[eval_name]['best_value'] = metric[best_name]
                                best_indicator[eval_name] = True 
                            else:
                                best_indicator[eval_name] = False 
                            best_step = metric_logger_dict[eval_name]['best_step']
                            LOGGER.info(f"======evaluation--{eval_name}====history best step: {best_step}=======\n")
                            LOGGER.info(metric_logger_dict[eval_name][str(best_step)])          
                
                model_saver.save(model, global_step, optimizer,best_indicator, run_cfg.save_best)
    

        if global_step >= run_cfg.num_train_steps:
            break
    pbar.close()






    
  
def test(model, test_loader, run_cfg):
    
    evaluate_fn = evaluation_registry[model.config.evaluation_type]
    eval_log = evaluate_fn(model, test_loader, run_cfg, global_step=0)
    if dist.get_rank()==0:  
        for task_name, val_log in eval_log.items():
            for eval_name, metric in val_log.items():
                eval_name = task_name +'_' +eval_name 
                # TB_LOGGER.log_scaler_dict({f"eval/{eval_name}/test_{k}": v
                #                 for k, v in metric.items() if not isinstance(v,str)})
                LOGGER.info(f"==== evaluation--{eval_name}========\n")
                LOGGER.info(metric)




def get_best_name(eval_name, metric):
    if eval_name.startswith('cap'):
        return 'CIDEr'
    elif eval_name.startswith('qa'):
        return 'accuracy'
    elif eval_name.startswith('ret'):
        if 'video_r1' in metric:
            return 'video_r1'
        if 'volume_T2D_r1' in metric:
            return 'volume_T2D_r1'
        if 'gramian_value' in metric:
            return 'gramian_value'
    elif eval_name.startswith('pt'):
        return None 
    else:
        raise NotImplementedError

