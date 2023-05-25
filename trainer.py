import os
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup
# from model import Ernie
from utils import MODEL_CLASSES, compute_metrics, get_intent_labels, get_slot_labels, multi_compute_metrics
logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.intent_label_lst = get_intent_labels(args)
        self.slot_label_lst = get_slot_labels(args)

        # if args.order_type:
        #     self.intent_label_lst[0]='O'

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = args.ignore_index

        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]

        self.config = self.config_class.from_pretrained(args.model_name_or_path, finetuning_task=args.task)
        self.model = self.model_class.from_pretrained(args.model_name_or_path,
                                                      config=self.config,
                                                      args=args,
                                                      intent_label_lst=self.intent_label_lst,
                                                      slot_label_lst=self.slot_label_lst)

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)
        max_slot_f1 = 0
        max_overall_acc = 0
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (
                        len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                    num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                intent_num_batch=[]
                for i in batch[3]:
                    intent_num_batch.append(torch.sum(i).item()-1)
                intent_num_batch=torch.tensor(intent_num_batch)
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'intent_label_ids': batch[3],
                          'slot_labels_ids': batch[4],
                          'intent_num_batch':intent_num_batch,
                          'train_or_val': 'train'
                          }
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]
                # if self.args.order_type:

                # outputs,intent_pred_index,_=self.model(**inputs)
                # test(outputs, inputs, self,intent_pred_index)

                # else:
                outputs = self.model(**inputs)
                # test(outputs, inputs, self)
                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        result=self.evaluate("test")
                        if result["sementic_frame_acc"]>max_overall_acc :
                            self.save_model()
                            max_overall_acc=result["sementic_frame_acc"]
                    #     self.save_model()
                    #     max_overall_acc=result["sementic_frame_acc"]
                    #     max_slot_f1=result["slot_f1"]
                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        self.save_model()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        intent_preds = None
        slot_preds = None
        out_intent_label_ids = None
        out_slot_labels_ids = None
        int_preds = None
        real_intent_acc = []
        pred_intent_acc = []
        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                intent_num_batch=[]
                for i in batch[3]:
                    intent_num_batch.append(torch.sum(i).item()-1)
                intent_num_batch=torch.tensor(intent_num_batch)
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'intent_label_ids': batch[3],
                          'slot_labels_ids': batch[4],
                          'intent_num_batch':intent_num_batch,
                          'train_or_val': 'val'
                          }
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]
                if self.args.order_type:
                    outputs, int_preds, intent_int,_ = self.model(**inputs)
                    # print('intent_int:',type(intent_int),intent_int[0])

                    # use int code
                    for i in intent_int:
                        temp = []
                        for j in i:
                            temp.append(self.intent_label_lst[j])
                        pred_intent_acc.append(temp)

                    for i in inputs['intent_label_ids'].detach().cpu().numpy():
                        temp = []
                        for j in range(len(i)):
                            if i[j] == 1:
                                temp.append(self.intent_label_lst[j])
                        real_intent_acc.append(temp)



                else:
                    outputs = self.model(**inputs)
                # outputs = self.model(**inputs)
                tmp_eval_loss, (intent_logits, slot_logits) = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Intent prediction
            if intent_preds is None:
                if self.args.order_type == False:
                    intent_preds = intent_logits.detach().cpu().numpy()
                else:
                    intent_preds = int_preds
                out_intent_label_ids = inputs['intent_label_ids'].detach().cpu().numpy()
            else:
                if self.args.order_type == False:
                    intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
                else:
                    intent_preds = np.append(intent_preds, int_preds.detach().cpu().numpy(), axis=0)

                out_intent_label_ids = np.append(
                    out_intent_label_ids, inputs['intent_label_ids'].detach().cpu().numpy(), axis=0)

            # Slot prediction
            if slot_preds is None:

                slot_preds = slot_logits.detach().cpu().numpy()

                out_slot_labels_ids = inputs["slot_labels_ids"].detach().cpu().numpy()
            else:

                slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)

                out_slot_labels_ids = np.append(out_slot_labels_ids, inputs["slot_labels_ids"].detach().cpu().numpy(),
                                                axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Intent result
        if self.args.order_type == False:
            intent_preds = np.argmax(intent_preds, axis=1)

        # else:
        #     #intent_preds = intent_preds.detach().cpu().numpy()
        #     intent_label_map = {i: label for i, label in enumerate(self.intent_label_lst)}
        #     out_intent_label_list = [[] for _ in range(out_intent_label_ids.shape[0])]
        #     intent_preds_list = [[] for _ in range(out_intent_label_ids.shape[0])]
        #     for i in range(intent_preds.shape[0]):
        #         for j in range(intent_preds.shape[1]):
        #             # if out_intent_label_list[i, j] != self.pad_token_label_id:
        #             out_intent_label_list[i].append(intent_label_map[out_intent_label_ids[i][j]])
        #             intent_preds_list[i].append(intent_label_map[intent_preds[i][j]])

        # Slot result
        slot_preds = np.argmax(slot_preds, axis=2)
        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

        for i in range(out_slot_labels_ids.shape[0]):
            for j in range(out_slot_labels_ids.shape[1]):
                if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                    out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])
        # print('intent_preds_list:',type(intent_preds_list),intent_preds_list[0])
        # print('intent_label:',type(out_intent_label_list),out_intent_label_list[0])
        # print('slot_preds_list:',type(slot_preds_list),slot_preds_list[0])
        # print('slot_label:',type(out_slot_label_list),out_slot_label_list[0])
        print('intent_preds:', type(intent_preds), intent_preds.dtype)
        print('out_intent_label_ids:', type(out_intent_label_ids), out_intent_label_ids.dtype)
        # print('pred_intent_acc:',type(pred_intent_acc),pred_intent_acc[0],pred_intent_acc.shape,pred_intent_acc.dtype)
        # print('real_intent_acc:',type(real_intent_acc),real_intent_acc[0],real_intent_acc.shape,real_intent_acc.dtype)

        # intent_preds=intent_preds.astype('int')
        # out_intent_label_ids=out_intent_label_ids.astype('int')
        # flag1=1
        # for i in intent_preds:
        #     for j in i:
        #         if j!=0 and j!=1:
        #             flag1=0
        #             print('intent_pred:',i)

        # flag2=1
        # for i in out_intent_label_ids:
        #     for j in i:
        #         if j!=0 and j!=1:
        #             flag2=0
        #             print('out_intent_label_ids:',i)

        # print('flag1:',flag1)
        # print('flag2:',flag2)

        # print('intent_preds:',intent_preds.dtype)
        # print('out_intent_label_ids:',out_intent_label_ids.dtype)
        if self.args.order_type == False:
            total_result = compute_metrics(intent_preds, out_intent_label_ids, slot_preds_list, out_slot_label_list)
        else:
            total_result = multi_compute_metrics(intent_preds, out_intent_label_ids, slot_preds_list,
                                                 out_slot_label_list, pred_intent_acc, real_intent_acc)
        results.update(total_result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = self.model_class.from_pretrained(self.args.model_dir,
                                                          args=self.args,
                                                          intent_label_lst=self.intent_label_lst,
                                                          slot_label_lst=self.slot_label_lst)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")

    def test_model(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        intent_preds = None
        slot_preds = None
        out_intent_label_ids = None
        out_slot_labels_ids = None
        intent_num=None
        int_preds = None
        real_intent_acc = []
        pred_intent_acc = []
        #        self.model.eval()
        num_indecs=None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                intent_num_batch=[]
                for i in batch[3]:
                    intent_num_batch.append(torch.sum(i).item()-1)
                intent_num_batch=torch.tensor(intent_num_batch)
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'intent_label_ids': batch[3],
                          'slot_labels_ids': batch[4],
                          'intent_num_batch':intent_num_batch,
                          'train_or_val': 'val'
                          }
                if num_indecs is None:
                    num_indecs = intent_num_batch.detach().cpu().numpy()+1
                else:
                    num_indecs = np.append(num_indecs, intent_num_batch.detach().cpu().numpy()+1, axis=0)

                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]
                if self.args.order_type:
                    outputs, int_preds, intent_int,pred_num = self.model(**inputs)
                    # print('intent_int:',type(intent_int),intent_int[0])

                    if intent_num is None:
                        intent_num = pred_num.detach().cpu().numpy()
                    else:
                        intent_num = np.append(intent_num, pred_num.detach().cpu().numpy(), axis=0)




                    # use int code
                    for i in intent_int:
                        temp = []
                        for j in i:
                            temp.append(self.intent_label_lst[j])
                        pred_intent_acc.append(temp)

                    for i in inputs['intent_label_ids'].detach().cpu().numpy():
                        temp = []
                        for j in range(len(i)):
                            if i[j] == 1:
                                temp.append(self.intent_label_lst[j])
                        real_intent_acc.append(temp)



                else:
                    outputs = self.model(**inputs)
                # outputs = self.model(**inputs)
                tmp_eval_loss, (intent_logits, slot_logits) = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Intent prediction
            if intent_preds is None:
                if self.args.order_type == False:
                    intent_preds = intent_logits.detach().cpu().numpy()
                else:
                    intent_preds = int_preds
                out_intent_label_ids = inputs['intent_label_ids'].detach().cpu().numpy()
            else:
                if self.args.order_type == False:
                    intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
                else:
                    intent_preds = np.append(intent_preds, int_preds.detach().cpu().numpy(), axis=0)

                out_intent_label_ids = np.append(
                    out_intent_label_ids, inputs['intent_label_ids'].detach().cpu().numpy(), axis=0)

            # Slot prediction
            if slot_preds is None:

                slot_preds = slot_logits.detach().cpu().numpy()

                out_slot_labels_ids = inputs["slot_labels_ids"].detach().cpu().numpy()
            else:

                slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)

                out_slot_labels_ids = np.append(out_slot_labels_ids, inputs["slot_labels_ids"].detach().cpu().numpy(),
                                                axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Intent result
        if self.args.order_type == False:
            intent_preds = np.argmax(intent_preds, axis=1)

        # else:
        #     #intent_preds = intent_preds.detach().cpu().numpy()
        #     intent_label_map = {i: label for i, label in enumerate(self.intent_label_lst)}
        #     out_intent_label_list = [[] for _ in range(out_intent_label_ids.shape[0])]
        #     intent_preds_list = [[] for _ in range(out_intent_label_ids.shape[0])]
        #     for i in range(intent_preds.shape[0]):
        #         for j in range(intent_preds.shape[1]):
        #             # if out_intent_label_list[i, j] != self.pad_token_label_id:
        #             out_intent_label_list[i].append(intent_label_map[out_intent_label_ids[i][j]])
        #             intent_preds_list[i].append(intent_label_map[intent_preds[i][j]])

        # Slot result
        slot_preds = np.argmax(slot_preds, axis=2)
        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

        for i in range(out_slot_labels_ids.shape[0]):
            for j in range(out_slot_labels_ids.shape[1]):
                if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                    out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])
        # print('intent_preds_list:',type(intent_preds_list),intent_preds_list[0])
        # print('intent_label:',type(out_intent_label_list),out_intent_label_list[0])
        # print('slot_preds_list:',type(slot_preds_list),slot_preds_list[0])
        # print('slot_label:',type(out_slot_label_list),out_slot_label_list[0])
        print('intent_preds:', type(intent_preds), intent_preds.dtype)
        print('out_intent_label_ids:', type(out_intent_label_ids), out_intent_label_ids.dtype)
        # print('pred_intent_acc:',type(pred_intent_acc),pred_intent_acc[0],pred_intent_acc.shape,pred_intent_acc.dtype)
        # print('real_intent_acc:',type(real_intent_acc),real_intent_acc[0],real_intent_acc.shape,real_intent_acc.dtype)


        sum_num=0
        sum_right_one=0
        sum_right_two=0
        sum_right_three=0
        intent_num=list(intent_num)
        num_indecs=list(num_indecs)
        for i in range(len(intent_num)):
            if intent_num[i]==num_indecs[i]:
                sum_num=sum_num+1
                if intent_num[i]==1:
                    sum_right_one=sum_right_one+1
                if intent_num[i]==2:
                    sum_right_two=sum_right_two+1
                if intent_num[i]==3:
                    sum_right_three=sum_right_three+1
        print(len(intent_num))
        print(sum_right_one)



        if self.args.order_type == False:
            total_result = compute_metrics(intent_preds, out_intent_label_ids, slot_preds_list, out_slot_label_list)
        else:
            total_result = multi_compute_metrics(intent_preds, out_intent_label_ids, slot_preds_list,
                                                 out_slot_label_list, pred_intent_acc, real_intent_acc)
        results.update(total_result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results


def test(outputs, inputs, self, int_preds=None):
    results = {}
    intent_preds = None
    slot_preds = None
    out_intent_label_ids = None
    out_slot_labels_ids = None

    self.model.eval()

    # outputs = self.model(**inputs)
    tmp_eval_loss, (intent_logits, slot_logits) = outputs[:2]
    # Intent prediction
    if intent_preds is None:
        if self.args.order_type == False:
            intent_preds = intent_logits.detach().cpu().numpy()
        else:
            intent_preds = int_preds
        out_intent_label_ids = inputs['intent_label_ids'].detach().cpu().numpy()
    else:
        if self.args.order_type == False:
            intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
        else:

            intent_preds = int_preds
        out_intent_label_ids = np.append(
            out_intent_label_ids, inputs['intent_label_ids'].detach().cpu().numpy(), axis=0)
    intent_preds = intent_preds.detach().cpu().numpy()
    intent_label_map = {i: label for i, label in enumerate(self.intent_label_lst)}
    out_intent_label_list = [[] for _ in range(int_preds.shape[0])]
    intent_preds_list = [[] for _ in range(int_preds.shape[0])]
    for i in range(int_preds.shape[0]):
        for j in range(int_preds.shape[1]):
            # if out_intent_label_list[i, j] != self.pad_token_label_id:
            out_intent_label_list[i].append(intent_label_map[out_intent_label_ids[i][j]])
            intent_preds_list[i].append(intent_label_map[intent_preds[i][j]])

    # Slot prediction
    if slot_preds is None:

        slot_preds = slot_logits.detach().cpu().numpy()

        out_slot_labels_ids = inputs["slot_labels_ids"].detach().cpu().numpy()
    else:

        slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)

        out_slot_labels_ids = np.append(out_slot_labels_ids, inputs["slot_labels_ids"].detach().cpu().numpy(), axis=0)

    # Intent result
    if self.args.order_type == False:
        intent_preds = np.argmax(intent_preds, axis=1)

    # Slot result
    slot_preds = np.argmax(slot_preds, axis=2)
    slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
    out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
    slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

    for i in range(out_slot_labels_ids.shape[0]):
        for j in range(out_slot_labels_ids.shape[1]):
            if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])
                slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])
    if self.args.order_type:
        total_result = multi_compute_metrics(intent_preds_list, out_intent_label_list, slot_preds_list,
                                             out_slot_label_list)
    else:
        total_result = compute_metrics(intent_preds, out_intent_label_ids, slot_preds_list, out_slot_label_list)
    results.update(total_result)

    logger.info("***** Eval results *****")
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results