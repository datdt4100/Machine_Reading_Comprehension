import json, collections, os, random, glob, math, string, re, torch
import numpy as np
import timeit
from tqdm import trange, tqdm_notebook as tqdm 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import WEIGHTS_NAME, BertConfig, BertForQuestionAnswering, BertTokenizerFast, BasicTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers.models.bert.tokenization_bert import whitespace_tokenize
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
from transformers import RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer
from transformers import XLMRobertaConfig, XLMRobertaForQuestionAnswering, XLMRobertaTokenizer
from transformers import AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer

import logging
import math
import re
import string

from utils import *
from dataset import *


logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    if args["n_gpu"] > 0:
        torch.cuda.manual_seed_all(args["seed"])
def evaluate(args, model, tokenizer, dataset, examples, features, prefix=""):
    
    if not os.path.exists(args["output_dir"]) and args["local_rank"] in [-1, 0]:
        os.makedirs(args["output_dir"])

    args["eval_batch_size"] = args["per_gpu_eval_batch_size"] * max(1, args["n_gpu"])

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args["eval_batch_size"])

    # multi-gpu evaluate
    if args["n_gpu"] > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    print("***** Running evaluation {} *****".format(prefix))
    print("  Num examples = %d" % len(dataset))
    print("  Batch size = %d" % args["eval_batch_size"])

    all_results = []
    start_time = timeit.default_timer()
    model.eval()

    for batch in tqdm(eval_dataloader):
        
        batch = tuple(t.to(args["device"]) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                'token_type_ids': None if args["model_type"] in ['xlm', 'roberta'] else batch[2]  # XLM don't use segment_ids
            }

            if args["model_type"] in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]

            example_indices = batch[3]

            # XLNet and XLM use more arguments for their predictions
            if args["model_type"] in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                # for lang_id-sensitive xlm models
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args["lang_id"]).to(args["device"])}
                    )

            outputs = model(**inputs, return_dict=False)

        for i, example_index in enumerate(example_indices):
            try:
              eval_feature = features[example_index.item()]
              unique_id = int(eval_feature.unique_id)
            except: print(example_index.item(), len(features))
            output = [to_list(output[i]) for output in outputs]

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    print(("  Evaluation done in total %f secs (%f sec per example)") % ( evalTime, evalTime / len(dataset)))

    # Compute predictions
    output_prediction_file = os.path.join(args["output_dir"], "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args["output_dir"], "nbest_predictions_{}.json".format(prefix))

    if args["version_2_with_negative"]:
        output_null_log_odds_file = os.path.join(args["output_dir"], "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    # XLNet and XLM use a more complex post-processing procedure
    if args["model_type"] in ["xlnet", "xlm"]:
        start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
        end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

        predictions = compute_predictions_log_probs(
            examples,
            features,
            all_results,
            args["n_best_size"],
            args["max_answer_length"],
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            start_n_top,
            end_n_top,
            args["version_2_with_negative"],
            tokenizer,
            args["verbose_logging"],
        )
    else:
        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            args["n_best_size"],
            args["max_answer_length"],
            args["do_lower_case"],
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            args["verbose_logging"],
            args["version_2_with_negative"],
            args["null_score_diff_threshold"],
            tokenizer, args["model_type"]
        )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    return results

def train(args,
          train_dataset,
          test_dataset,
          model,
          tokenizer,
          tb_writer = None):
    """
    Train the model.
    """

    #DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)

    args["train_batch_size"] = args["per_gpu_train_batch_size"] * max(1, args["n_gpu"])
    train_sampler = RandomSampler(train_dataset) if args["local_rank"] == -1 else DistributedSampler(train_dataset)
    #train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,  sampler=train_sampler, batch_size=args["train_batch_size"])
    #train_dataloader = DataLoader(train_dataset,  shuffle=False, sampler=None, batch_size=args["train_batch_size"])

    if args["max_steps"] > 0:
        t_total = args["max_steps"]
        args["num_train_epochs"] = args["max_steps"] // (len(train_dataloader) // args["gradient_accumulation_steps"]) + 1
    else:
        t_total = len(train_dataloader) // args["gradient_accumulation_steps"] * args["num_train_epochs"]

    # Prepare optimizer and scheduler (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args["weight_decay"]},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args["learning_rate"],
                      eps=args["adam_epsilon"])
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                     num_warmup_steps=args["warmup_steps"],
                                     num_training_steps=t_total)
    # Multiple GPU training
    if args["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training
    if args["local_rank"] != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args["local_rank"]],
                                                          output_device=args["local_rank"],
                                                          find_unused_parameters=True)

    # Training
    print("***** Running training *****")
    print("  Num examples = %d" % len(train_dataset))
    print("  Num Epochs = %d" % args["num_train_epochs"])
    print("  Instantaneous batch size per GPU = %d", args["per_gpu_train_batch_size"])
    print("  Total train batch size (w. parallel, distributed & accumulation) = %d" %
                   (args["train_batch_size"] * args["gradient_accumulation_steps"] * (torch.distributed.get_world_size() if args["local_rank"] != -1 else 1)))
    print("  Gradient Accumulation steps = %d" % args["gradient_accumulation_steps"])
    print("  Total optimization steps = %d" % t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    # Added here for reproductibility (even between python 2 and 3)
    set_seed(args)
    epochs = int(args["num_train_epochs"])
    best_score = 0
    for epoch in range(epochs):
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        mloss = torch.zeros(1, device=args["device"])  # mean losses
        for step, batch in pbar:
            model.train()
            batch = tuple(t.to(args["device"]) for t in batch)
            inputs = {'input_ids':       batch[0],
                      'attention_mask':  batch[1],
                      'token_type_ids':  None if args["model_type"] in ['xlm', 'roberta'] else batch[2],
                      'start_positions': batch[3],
                      'end_positions':   batch[4]}

            if args["model_type"] in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[5],
                               'p_mask': batch[6]})
            outputs = model(**inputs)
            # Model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args["n_gpu"] > 1:
                # `mean()` to average on multi-gpu parallel (not distributed) training
                loss = loss.mean()
            if args["gradient_accumulation_steps"] > 1:
                loss = loss / args["gradient_accumulation_steps"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args["max_grad_norm"])

            tr_loss += loss.item()
            if (step + 1) % args["gradient_accumulation_steps"] == 0:
                optimizer.step()
                # Update learning rate schedule
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args["local_rank"] in [-1, 0] and args["logging_steps"] > 0 and global_step % args["logging_steps"] == 0:
                    # Log metrics
                    # Only evaluate when single GPU otherwise metrics may not average well
                    tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args["logging_steps"], global_step)
                    logging_loss = tr_loss

                if args["local_rank"] in [-1, 0] and args["save_steps"] > 0 and global_step % args["save_steps"] == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args["output_dir"], 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    # print("Saving model checkpoint to %s", output_dir)

            if args["max_steps"] > 0 and global_step > args["max_steps"]:
                epoch_iterator.close()
                break
            if args["local_rank"] in [-1, 0]:
                mloss = (mloss * step + loss) / (step + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('Epoch %10s --- mem: %8s --- loss: %10.4g') % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss)
                pbar.set_description(s)
        # Eval
        results = evaluate(args, model, tokenizer, test_dataset[0], test_dataset[1], test_dataset[2])
        eval_str = ''
        for key, value in results.items():
            eval_str += ('%30s' + '%10.4g' + '\n') % (key, value)
        logger.info(eval_str)
        for key, value in results.items():
            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

        avg_score = results['exact'] * results['f1'] * 2 / (results['exact'] + results['f1'])
        if avg_score > best_score:
            output_dir = os.path.join(args["output_dir"], 'checkpoint-best')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            print("Saving model checkpoint to %s" % output_dir)
        if args["max_steps"] > 0 and global_step > args["max_steps"]:
            train_iterator.close()
            break

    #if args["local_rank"] in [-1, 0]:
    #    tb_writer.close()

    return global_step, tr_loss / global_step


MODEL_CLASSES = {
    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizerFast),
    #'bert': (BertConfig, BertForQuestionAnswering, BertTokenizerFast),
    #'roberta': (RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer)
    'roberta': (XLMRobertaConfig, XLMRobertaForQuestionAnswering, XLMRobertaTokenizer)
}

if __name__ == '__main__':
    args = {
        "project":'MRC-VN',
        "local_rank": -1,
        "model_type": "roberta",
        "config_name": r"D:\NLP_project\vireader\models\xlmr-finetuned-viquad_k5\checkpoint-best\config.json", 
        #"model_name_or_path": "bert-base-multilingual-cased", 
        "model_name_or_path": r"D:\NLP_project\vireader\models\xlmr-finetuned-viquad_k5\checkpoint-best\pytorch_model.bin",
        "base_model_name": "xlm-roberta-base",
        "tokenizer_name": "xlm-roberta-base",
        "max_seq_length": 400,  
        "overwrite_cache": False,
        #"do_lower_case": True,
        "do_lower_case": False,
        "do_train": True,
        "output_dir": "models/xlmr-finetuned-viquad_k5", 
        "version_2_with_negative": True,
        "doc_stride": 128,
        "max_query_length": 128,
        "train_file": r"D:/NLP_project/data/SQuA2.0/train-v2.0-vi.json",
        "predict_file":r"D:/NLP_project/data/SQuA2.0/test-v2.0-vi.json",
        "per_gpu_train_batch_size": 8, 
        "max_steps": -1,
        "num_train_epochs": 6,
        "learning_rate": 2e-5,   
        "adam_epsilon": 1e-8,
        "warmup_steps": 100,   
        "no_cuda": False,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "weight_decay": 0.0,  
        "save_steps": 2000,
        "logging_steps": 100,
        "seed": 42,
        "do_eval": True,
        "eval_all_checkpoints": True,
        "eval_batch_size": 16,
        "per_gpu_eval_batch_size": 16,  
        "n_best_size": 20,
        "max_answer_length": 500,
        "null_score_diff_threshold": 0.0,
        "verbose_logging": False,
        "version_2_with_negative": True
        
    }

    if args["local_rank"] == -1 or args["no_cuda"]:
        device = torch.device("cuda" if torch.cuda.is_available() and not args["no_cuda"] else "cpu")
        args["n_gpu"] = torch.cuda.device_count()
    else:  
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args["local_rank"])
        device = torch.device("cuda", args["local_rank"])
        torch.distributed.init_process_group(backend='nccl')
        args["n_gpu"] = 1
    set_logging(args["local_rank"])
    tb_writer = None  # init loggers
    if args["local_rank"] in [-1, 0]:
        prefix = colorstr('tensorboard: ')
        logger.info(f"{prefix}Start with 'tensorboard --logdir {args['project']}', view at http://localhost:6006/")
        tb_writer = SummaryWriter(args["output_dir"])  # Tensorboard
    init_seeds(args["seed"])
    args["device"] = device
        
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args["model_type"]]
    config = config_class.from_pretrained(args["config_name"] if args["config_name"] else args["model_name_or_path"])
    tokenizer = tokenizer_class.from_pretrained(args["tokenizer_name"] if args["tokenizer_name"] else args["model_name_or_path"], do_lower_case=args["do_lower_case"], strip_accents=False, add_special_tokens=False)
    model = model_class.from_pretrained(args["model_name_or_path"], from_tf=bool('.ckpt' in args["model_name_or_path"]), config=config)

    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False, model_type=args["model_type"])
    test_dataset = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True, model_type=args["model_type"])

    # Make sure only the first process in distributed training will download model & vocab
    if args["local_rank"] == 0:
        torch.distributed.barrier()  

    model.to(args["device"])

    print("Training/evaluation parameters %s" % args)
        
    # Training
    if args["do_train"]:
        global_step, tr_loss = train(args, 
                                    train_dataset,
                                    test_dataset,
                                    model, 
                                    tokenizer,
                                    tb_writer)
        print((" global_step = %s, average loss = %s") %( global_step, tr_loss))

    # Save the trained model and the tokenizer
    if args["do_train"] and (args["local_rank"] == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args["output_dir"]) and args["local_rank"] in [-1, 0]:
            os.makedirs(args["output_dir"])

        print("Saving model checkpoint to %s" % args["output_dir"])
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, 'module') else model  
        model_to_save.save_pretrained(args["output_dir"])
        tokenizer.save_pretrained(args["output_dir"])

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args["output_dir"], 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args["output_dir"])
        tokenizer = tokenizer_class.from_pretrained(args["output_dir"], do_lower_case=args["do_lower_case"])
        model.to(device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args["do_eval"] and args["local_rank"] in [-1, 0]:
        checkpoints = [args["output_dir"]]
        if args["eval_all_checkpoints"]:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args["output_dir"] + '/**/' + WEIGHTS_NAME, recursive=True)))

        print("Evaluate the following checkpoints: %s" % checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(device)

            # Evaluate
            result = evaluate(args, model, tokenizer, test_dataset[0], test_dataset[1], test_dataset[2], prefix=global_step)

            result = dict((k + ('_{}'.format(global_step) if global_step else ''), v) for k, v in result.items())
            results.update(result)

    print("Results: {}".format(results))