import argparse
import logging
# import wandb
import torch
from collections import defaultdict
from torch.utils.data import DataLoader
from transformers import AutoProcessor, get_linear_schedule_with_warmup
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from tqdm.auto import tqdm
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score
import re
from tqdm import tqdm

from models_astro_ultra_qwen2 import AstroQwen2VLForConditionalGeneration
from dataset_ultra_qwen2vl import Qwen2VLTrainingDataset, collate_fn,  Qwen2VLRegressionEvaluationDataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)

logger = get_logger(__name__)

def prepare_model_for_training(model):
    """冻结基础模型参数,只训练新增参数"""
    # 首先冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
        
    # 分别处理Module和Parameter
    trainable_modules = [
        model.spec_projector,
        model.struc_projector,
        model.spec_norm,
        model.struc_norm,
        model.num_head
    ]
    
    trainable_parameters = [
        model.spec_scale,
        model.struc_scale,
        model.lm_weight,
        model.regression_weight
    ]
    
    # 添加每一层的expert参数
    for layer in model.model.layers:
        if hasattr(layer, 'moe'):  # 确保layer有moe属性
            trainable_modules.extend([
                layer.moe.router,
                layer.moe.experts[0],  # EuclideanFFN
                layer.moe.experts[1],  # HyperbolicFFN
                layer.moe.experts[2],  # SphericalFFN
            ])
            trainable_parameters.append(layer.moe.temperature)
    
    # 设置Module的参数为可训练
    for module in trainable_modules:
        for param in module.parameters():
            param.requires_grad = True
            
    # 设置Parameter为可训练
    for param in trainable_parameters:
        param.requires_grad = True
            
    # 打印可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {trainable_params}")
    
    return model

def extract_number(text: str) -> float:
    try:
        match = re.search(r'-?\d*\.?\d+', text)
        if match:
            return float(match.group())
        return float('nan')
    except:
        return float('nan')

def evaluate_regression(model, eval_dataloader, accelerator, args, global_step):
    """评估回归任务"""
    model.eval()
    task_metrics = {}
    task_predictions = defaultdict(list)
    task_labels = defaultdict(list)
    
    print(f"Evaluating regression tasks, total batches: {len(eval_dataloader)}")
    for batch in eval_dataloader:
        with torch.no_grad():
            batch_tasks = batch['task_types']
            # print(batch["text_sequences"][0][0])
            
            # 一次只生成一个token
            for i in range(5):  # 最多生成5个token
                if i == 0:
                    # 第一个token
                    outputs = model.generate(
                        **batch["processed_inputs"],
                        max_new_tokens=1,
                        do_sample=False,
                        output_hidden_states=True,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                    current_ids = outputs.sequences
                    current_hidden = outputs.hidden_states[-1][-1]  # 最后一层的最后一个token的hidden states
                else:
                    # 后续token，需要使用之前的结果作为输入
                    new_inputs = batch["processed_inputs"].copy()
                    new_inputs["input_ids"] = current_ids
                    new_inputs["attention_mask"] = torch.ones_like(current_ids)
                    
                    outputs = model.generate(
                        **new_inputs,
                        max_new_tokens=1,
                        do_sample=False,
                        output_hidden_states=True,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                    current_ids = outputs.sequences
                    current_hidden = outputs.hidden_states[-1][-1]
                
                # 检查每个样本是否生成了" num"
                for batch_idx, (ids, task) in enumerate(zip(current_ids, batch_tasks)):
                    last_token = ids[-1].item()
                    if last_token == model.num_token_id:
                        # 如果是" num"，立即使用regression head进行预测
                        pred = model.predict_number(
                            input_ids=ids.unsqueeze(0),
                            hidden_states=current_hidden[batch_idx].unsqueeze(0),
                            spec_features=batch["processed_inputs"].get('spec_features')[batch_idx:batch_idx+1] if 'spec_features' in batch["processed_inputs"] else None,
                            euc_features=batch["processed_inputs"].get('euc_features')[batch_idx:batch_idx+1] if 'euc_features' in batch["processed_inputs"] else None,
                            hyp_features=batch["processed_inputs"].get('hyp_features')[batch_idx:batch_idx+1] if 'hyp_features' in batch["processed_inputs"] else None,
                            sph_features=batch["processed_inputs"].get('sph_features')[batch_idx:batch_idx+1] if 'sph_features' in batch["processed_inputs"] else None
                        )
                        task_predictions[task].append(pred.item())
                        task_labels[task].append(batch['answers'][batch_idx])
                        
                        # 从当前batch中移除已处理的样本
                        mask = torch.ones(len(batch_tasks), dtype=torch.bool)
                        mask[batch_idx] = False
                        batch_tasks = [t for t, m in zip(batch_tasks, mask) if m]
                        for k in batch["processed_inputs"]:
                            if isinstance(batch["processed_inputs"][k], torch.Tensor):
                                batch["processed_inputs"][k] = batch["processed_inputs"][k][mask]
                
                # 如果所有样本都已处理完，提前退出
                if len(batch_tasks) == 0:
                    break
            
            # 对于未生成" num"的样本，添加nan
            for task in batch_tasks:
                task_predictions[task].append(float('nan'))
                task_labels[task].append(batch['answers'][0])  # 注意这里可能需要调整
    
    # 计算每个任务的指标
    for task in task_predictions.keys():
        predictions = np.array(task_predictions[task])
        labels = np.array(task_labels[task])
        
        # 计算指标
        mse = np.mean((predictions - labels) ** 2)
        flag = np.any(np.isnan(predictions))
        r2 = 0 if flag else r2_score(labels, predictions)
        
        print(f"\nTask: {task}")
        print(f"Samples: {len(predictions)}")
        print(f"Predictions (first 3): {predictions[:3]}")
        print(f"Labels (first 3): {labels[:3]}")
        print(f"MSE: {mse:.6f}, R2: {r2:.6f}")
        
        task_metrics[task] = {
            'mse': mse,
            'r2': r2
        }
        
        # # 记录到tensorboard
        # args.writer.add_scalar(f'{task}/mse', mse, global_step)
        # args.writer.add_scalar(f'{task}/r2', r2, global_step)

        # if accelerator.is_main_process:
        #     wandb.log({
        #         f'{task}/mse': mse,
        #         f'{task}/r2': r2
        #     }, step=global_step)
    
    return task_metrics

def train():
    # torch.multiprocessing.set_start_method('spawn')
    #args
    parser = argparse.ArgumentParser()
    # Model and data arguments
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_regression_data", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--template_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    
    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    
    # Evaluation and logging
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=1000)
    
    
    parser.add_argument("--regression_eval_dir", type=str, required=True,
                       help="Directory containing split regression evaluation files")
    parser.add_argument("--eval_regression_tasks", nargs='+', 
                       help="Specific regression tasks to evaluate")
    parser.add_argument("--samples_per_regression_task", type=int, default=None,
                       help="Number of samples to evaluate per regression task")
    
    args = parser.parse_args(args=['--model_path', '/kaggle/working/models/Qwen2-VL-2B-Instruct',
                                   '--train_regression_data', '/kaggle/working/datasets/aaaaaaa/kaggle_datasets/train_add_feature.hdf5',
                                   '--regression_eval_dir', '/kaggle/working/datasets/aaaaaaa/kaggle_datasets',
                                   '--eval_regression_tasks', 'task1_redshift',
                                   '--image_dir', '/kaggle/working/datasets/aaaaaaa/kaggle_datasets/images',
                                   '--template_path', 'template_ultra_qwen2vl_classification.json',
                                   '--output_dir', '/kaggle/working/outputs',
                                   '--per_device_train_batch_size', '1',
                                   '--per_device_eval_batch_size', '1',
                                   '--gradient_accumulation_steps', '1',
                                   '--eval_steps', '1000',
                                   '--save_steps', '10000',
                                   '--samples_per_regression_task', '5',
                                   '--num_train_epochs', '3'
                                   ])
    
    # 构建数据字典
    args.train_data = {
        "regression": args.train_regression_data
        # "classification": args.train_classification_data
    }
    
    # 初始化accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs]
    )
    accelerator.print(f'device {str(accelerator.device)} is used!')
    
    # 设置随机种子
    set_seed(args.seed)
    
    # # 初始化tensorboard
    # if accelerator.is_main_process:
    #     args.writer = SummaryWriter(args.output_dir)
    #     wandb.login(key="c3fc632dc58c30c780f159d673f9ba5d39380b5e")
    #     wandb.init(project="galaxywalker", mode="offline")
    
    # 加载模型和数据
    min_pixels = 110*110*3
    max_pixels = 144*144*3
    processor = AutoProcessor.from_pretrained(args.model_path,min_pixels=min_pixels, max_pixels=max_pixels)
    args.processor = processor
    model = AstroQwen2VLForConditionalGeneration.from_pretrained(args.model_path,
                                                                 device_map="auto",
                                                                 low_cpu_mem_usage=True)
    model = prepare_model_for_training(model)

    print("准备数据集……")
    
    train_dataset = Qwen2VLTrainingDataset(
        hdf5_paths=args.train_data,
        image_dir=args.image_dir,
        template_path=args.template_path,
        processor=processor
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    eval_regression_dataset = Qwen2VLRegressionEvaluationDataset(
        eval_dir=args.regression_eval_dir,
        image_dir=args.image_dir,
        template_path=args.template_path,
        processor=processor,
        max_samples_per_task=args.samples_per_regression_task,
        selected_tasks=args.eval_regression_tasks
    )
    
    print("开始创建dataloader……")
    
    # 创建两个dataloader
    eval_regression_dataloader = DataLoader(
        eval_regression_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=collate_fn
    )
    
    # 准备优化器和学习率调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    num_update_steps = len(train_dataloader) * args.num_train_epochs
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_update_steps
    )
    
    # 准备训练
    model, optimizer, train_dataloader, eval_regression_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader,  eval_regression_dataloader
    )
    
    # 训练循环
    global_step = 0
    best_metrics = {
        'mse': float('inf'),
        'r2': float('-inf'),
        'accuracy': 0,
        'f1': 0
    }
    
    print("开始训练……")
    
    for epoch in range(args.num_train_epochs):
        model.train()
        
        for step, batch in enumerate(tqdm(train_dataloader)):
            with accelerator.accumulate(model):
                outputs = model(**batch["processed_inputs"], return_dict=True)
                loss = outputs.loss
                # print(loss)
                accelerator.backward(loss)

                # Clean up memory
                if 'outputs' in locals():
                    if hasattr(outputs, 'hidden_states'):
                        del outputs.hidden_states
                    if hasattr(outputs, 'attentions'):
                        del outputs.attentions
                    if hasattr(outputs, 'router_logits'):
                        del outputs.router_logits
                    if hasattr(outputs, 'past_key_values'):
                        del outputs.past_key_values
                
                # Clean up feature embeddings
                for tensor_name in ['spec_embeds', 'euc_embeds', 'hyp_embeds', 'sph_embeds', 'image_embeds']:
                    if tensor_name in locals():
                        del locals()[tensor_name]
                
                # Clean up masks
                for mask_name in ['spec_mask', 'euc_mask', 'hyp_mask', 'sph_mask', 'image_mask']:
                    if mask_name in locals():
                        del locals()[mask_name]
                
                # Clean up other intermediate tensors
                for tensor_name in ['shift_logits', 'shift_labels', 'num_mask', 'num_logits', 'regression_value']:
                    if tensor_name in locals():
                        del locals()[tensor_name]
                
                if step % args.gradient_accumulation_steps == 0:
                    torch.cuda.empty_cache()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    # # 记录训练loss
                    # if global_step % args.logging_steps == 0:
                        # args.writer.add_scalar('train/loss', loss.item(), global_step)
                        # if accelerator.is_main_process:
                        #     wandb.log({"train/loss": loss.item()}, step=global_step)
                        
                    # 评估
                  
                    if global_step % args.eval_steps == 0:
                        reg_metrics = evaluate_regression(
                            model, eval_regression_dataloader, accelerator, args, global_step
                        )
                        
                        # 更新最佳指标并保存模型
                        avg_mse = np.mean([m['mse'] for m in reg_metrics.values()])
                        avg_r2 = np.mean([m['r2'] for m in reg_metrics.values()])
                        
                        if avg_mse < best_metrics['mse']:
                            best_metrics['mse'] = avg_mse
                            accelerator.save_state(f"{args.output_dir}/best_mse")
                            
                        if avg_r2 > best_metrics['r2']:
                            best_metrics['r2'] = avg_r2
                            accelerator.save_state(f"{args.output_dir}/best_r2")
                    
                    # 定期保存checkpoint
                    if global_step % args.save_steps == 0:
                        accelerator.save_state(
                            f"{args.output_dir}/checkpoint-{global_step}"
                        )
    
    # 保存最终模型
    accelerator.wait_for_everyone()
    accelerator.save_state(f"{args.output_dir}/final")
    
    # if accelerator.is_main_process:
    #     args.writer.close()
    #     wandb.finish()

if __name__ == "__main__":
    train()