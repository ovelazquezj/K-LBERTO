# -*- encoding:utf-8 -*-
"""
  This script provides an K-BERT example for NER.
  
  FIXES APLICADOS (2026-01-03):
  - Bug 1: Eliminado import innecesario de calculate_ner_metrics
  - Bug 2: Eliminada inicialización innecesaria tokens, labels = [], []
  - Bug 3-4: Eliminados acumuladores all_preds/all_golds problemáticos
  - Bug 5: Eliminada concatenación errónea y llamada a calculate_ner_metrics
  - Bug 6: Agregado epoch en evaluate() durante training
  - Logging funcional con métricas overall (PER/LOC/ORG pendientes)
"""
import os
import random
import argparse
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from uer.model_builder import build_model
from uer.utils.config import load_hyperparam
from uer.utils.optimizers import  BertAdam
from uer.utils.constants import *
from uer.utils.vocab import Vocab
from uer.utils.seed import set_seed
from uer.model_saver import save_model
import numpy as np

from brain import KnowledgeGraph
from utils.metrics_logger import MetricsLogger


class BertTagger(nn.Module):
    def __init__(self, args, model):
        super(BertTagger, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.target = model.target
        self.labels_num = args.labels_num
        self.output_layer = nn.Linear(args.hidden_size, self.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src, label, mask, pos=None, vm=None):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size x seq_length]
            mask: [batch_size x seq_length]
        Returns:
            loss: Sequence labeling loss.
            correct: Number of labels that are predicted correctly.
            predict: Predicted label.
            label: Gold label.
        """
        # Embedding.
        emb = self.embedding(src, mask, pos)
        # Encoder.
        output = self.encoder(emb, mask, vm)
        # Target.
        output = self.output_layer(output)

        output = output.contiguous().view(-1, self.labels_num)
        output = self.softmax(output)

        label = label.contiguous().view(-1,1)
        label_mask = (label > 0).float().to(torch.device(label.device))
        one_hot = torch.zeros(label_mask.size(0),  self.labels_num). \
                  to(torch.device(label.device)). \
                  scatter_(1, label, 1.0)

        numerator = -torch.sum(output * one_hot, 1)
        label_mask = label_mask.contiguous().view(-1)
        label = label.contiguous().view(-1)
        numerator = torch.sum(label_mask * numerator)
        denominator = torch.sum(label_mask) + 1e-6
        loss = numerator / denominator
        predict = output.argmax(dim=-1)
        correct = torch.sum(
            label_mask * (predict.eq(label)).float()
        )

        return loss, correct, predict, label


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="./models/tagger_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", default="./models/google_vocab.txt", type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path of the trainset.")
    parser.add_argument("--dev_path", type=str, required=True,
                        help="Path of the devset.")
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path of the testset.")
    parser.add_argument("--config_path", default="./models/google_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch_size.")
    parser.add_argument("--seq_length", default=256, type=int,
                        help="Sequence length.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                                   "cnn", "gatedcnn", "attn", \
                                                   "rcnn", "crnn", "gpt", "bilstm"], \
                                                   default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")

    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=5,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    # kg
    parser.add_argument("--kg_name", required=True, help="KG name or path")

    # Logging options
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Experiment name for logging (e.g., D500_sqrt_seed42)")
    parser.add_argument("--output_dir", type=str, default="resultados",
                        help="Output directory for logs and results")


    args = parser.parse_args()

    # Load the hyperparameters of the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Initialize metrics logger
    if hasattr(args, 'experiment_name') and args.experiment_name:
        logger = MetricsLogger(
            output_dir=args.output_dir if hasattr(args, 'output_dir') else 'resultados',
            experiment_name=args.experiment_name
        )
        # Save experiment config
        config_dict = {
            'train_path': args.train_path,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'epochs_num': args.epochs_num,
            'seq_length': args.seq_length,
            'seed': args.seed,
            'kg_name': args.kg_name
        }
        logger.save_config(config_dict)
        print(f"[Logger] Initialized: {logger.experiment_name}")
    else:
        logger = None
        print("[Logger] No experiment_name provided - logging disabled")


    labels_map = {"[PAD]": 0, "[ENT]": 1}
    begin_ids = []

    # Find tagging labels
    with open(args.train_path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                continue
            labels = line.strip().split("\t")[0].split()
            for l in labels:
                if l not in labels_map:
                    if l.startswith("B") or l.startswith("S"):
                        begin_ids.append(len(labels_map))
                    labels_map[l] = len(labels_map)

    print("Labels: ", labels_map)
    args.labels_num = len(labels_map)

    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build knowledge graph.
    if args.kg_name == 'none':
        spo_files = []
    else:
        spo_files = [args.kg_name]
    kg = KnowledgeGraph(spo_files=spo_files, predicate=False)

    # Build bert model.
    # A pseudo target is added.
    args.target = "bert"
    model = build_model(args)

    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)

    # Build sequence labeling model.
    model = BertTagger(args, model)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model = model.to(device)

    # Datset loader.
    def batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vm_ids, tag_ids):
        instances_num = input_ids.size()[0]
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i*batch_size: (i+1)*batch_size, :]
            label_ids_batch = label_ids[i*batch_size: (i+1)*batch_size, :]
            mask_ids_batch = mask_ids[i*batch_size: (i+1)*batch_size, :]
            pos_ids_batch = pos_ids[i*batch_size: (i+1)*batch_size, :]
            vm_ids_batch = vm_ids[i*batch_size: (i+1)*batch_size, :, :]
            tag_ids_batch = tag_ids[i*batch_size: (i+1)*batch_size, :]
            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch, tag_ids_batch
        if instances_num > instances_num // batch_size * batch_size:
            input_ids_batch = input_ids[instances_num//batch_size*batch_size:, :]
            label_ids_batch = label_ids[instances_num//batch_size*batch_size:, :]
            mask_ids_batch = mask_ids[instances_num//batch_size*batch_size:, :]
            pos_ids_batch = pos_ids[instances_num//batch_size*batch_size:, :]
            vm_ids_batch = vm_ids[instances_num//batch_size*batch_size:, :, :]
            tag_ids_batch = tag_ids[instances_num//batch_size*batch_size:, :]
            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch, tag_ids_batch

    # Read dataset.
    def read_dataset(path):
        dataset = []
        with open(path, mode="r", encoding="utf-8") as f:
            f.readline()
            for line_id, line in enumerate(f):
                labels, tokens = line.strip().split("\t")

                text = tokens
                tokens, pos, vm, tag = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length)
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")
                tag = tag[0]

                tokens = [vocab.get(t) for t in tokens]
                labels = [labels_map[l] for l in labels.split(" ")]
                mask = [1] * len(tokens)

                new_labels = []
                j = 0
                for i in range(len(tokens)):
                    if tag[i] == 0 and tokens[i] != vocab.get("[PAD]"):
                        new_labels.append(labels[j])
                        j += 1
                    elif tag[i] == 1 and tokens[i] != vocab.get("[PAD]"):
                        new_labels.append(labels_map['[ENT]'])
                    else:
                        new_labels.append(labels_map[PAD_TOKEN])

                dataset.append([tokens, new_labels, mask, pos, vm, tag])

        return dataset

    # Evaluation function.
    def evaluate(args, is_test, epoch=0):
        if is_test:
            dataset = read_dataset(args.test_path)
        else:
            dataset = read_dataset(args.dev_path)

        input_ids = torch.LongTensor([sample[0] for sample in dataset])
        label_ids = torch.LongTensor([sample[1] for sample in dataset])
        mask_ids = torch.LongTensor([sample[2] for sample in dataset])
        pos_ids = torch.LongTensor([sample[3] for sample in dataset])
        vm_ids = torch.BoolTensor([sample[4] for sample in dataset])
        tag_ids = torch.LongTensor([sample[5] for sample in dataset])

        instances_num = input_ids.size(0)
        batch_size = args.batch_size

        if is_test:
            print("Batch size: ", batch_size)
            print("The number of test instances:", instances_num)

        # Overall counters
        correct = 0
        gold_entities_num = 0
        pred_entities_num = 0

        # Counters por tipo de entidad
        stats_per_type = {
            'PER': {'tp': 0, 'fp': 0, 'fn': 0},
            'LOC': {'tp': 0, 'fp': 0, 'fn': 0},
            'ORG': {'tp': 0, 'fp': 0, 'fn': 0}
        }

        # Crear mapa inverso de labels
        id2label = {v: k for k, v in labels_map.items()}

        confusion = torch.zeros(len(labels_map), len(labels_map), dtype=torch.long)

        model.eval()

        for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch, tag_ids_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vm_ids, tag_ids)):

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            tag_ids_batch = tag_ids_batch.to(device)
            vm_ids_batch = vm_ids_batch.long().to(device)

            loss, _, pred, gold = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch)

            # Procesar batch como antes para métricas overall
            for j in range(gold.size()[0]):
                if gold[j].item() in begin_ids:
                    gold_entities_num += 1

            for j in range(pred.size()[0]):
                if pred[j].item() in begin_ids and gold[j].item() != labels_map["[PAD]"]:
                    pred_entities_num += 1

            pred_entities_pos = []
            gold_entities_pos = []
            start, end = 0, 0

            # Extraer entidades gold con tipo
            gold_entities_typed = []
            for j in range(gold.size()[0]):
                if gold[j].item() in begin_ids:
                    start = j
                    label_name = id2label.get(gold[j].item(), 'O')
                    entity_type = label_name.split('-')[1] if '-' in label_name else 'UNK'
                    
                    for k in range(j+1, gold.size()[0]):
                        if gold[k].item() == labels_map['[ENT]']:
                            continue
                        if gold[k].item() == labels_map["[PAD]"] or gold[k].item() == labels_map["O"] or gold[k].item() in begin_ids:
                            end = k - 1
                            break
                    else:
                        end = gold.size()[0] - 1
                    
                    gold_entities_pos.append((start, end))
                    gold_entities_typed.append({'start': start, 'end': end, 'type': entity_type})

            # Extraer entidades pred con tipo
            pred_entities_typed = []
            for j in range(pred.size()[0]):
                if pred[j].item() in begin_ids and gold[j].item() != labels_map["[PAD]"] and gold[j].item() != labels_map["[ENT]"]:
                    start = j
                    label_name = id2label.get(pred[j].item(), 'O')
                    entity_type = label_name.split('-')[1] if '-' in label_name else 'UNK'
                    
                    for k in range(j+1, pred.size()[0]):
                        if gold[k].item() == labels_map['[ENT]']:
                            continue
                        if pred[k].item() == labels_map["[PAD]"] or pred[k].item() == labels_map["O"] or pred[k].item() in begin_ids:
                            end = k - 1
                            break
                    else:
                        end = pred.size()[0] - 1
                    
                    pred_entities_pos.append((start, end))
                    pred_entities_typed.append({'start': start, 'end': end, 'type': entity_type})

            # Calcular correct para overall
            for entity in pred_entities_pos:
                if entity not in gold_entities_pos:
                    continue
                else:
                    correct += 1

            # Calcular TP/FP/FN por tipo
            for ent_type in ['PER', 'LOC', 'ORG']:
                # Entidades de este tipo
                pred_type = [e for e in pred_entities_typed if e['type'] == ent_type]
                gold_type = [e for e in gold_entities_typed if e['type'] == ent_type]
                
                # Convertir a spans para comparación
                pred_spans = set((e['start'], e['end']) for e in pred_type)
                gold_spans = set((e['start'], e['end']) for e in gold_type)
                
                # TP: spans que están en ambos
                tp = len(pred_spans & gold_spans)
                # FP: spans predichos que no están en gold
                fp = len(pred_spans - gold_spans)
                # FN: spans gold que no fueron predichos
                fn = len(gold_spans - pred_spans)
                
                stats_per_type[ent_type]['tp'] += tp
                stats_per_type[ent_type]['fp'] += fp
                stats_per_type[ent_type]['fn'] += fn

        # Calcular métricas overall
        print("Report precision, recall, and f1:")
        p = correct/pred_entities_num if pred_entities_num > 0 else 0.0
        r = correct/gold_entities_num if gold_entities_num > 0 else 0.0
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0
        print("{:.3f}, {:.3f}, {:.3f}".format(p,r,f1))

        # Calcular métricas por tipo
        metrics_by_type = {}
        for ent_type in ['PER', 'LOC', 'ORG']:
            tp = stats_per_type[ent_type]['tp']
            fp = stats_per_type[ent_type]['fp']
            fn = stats_per_type[ent_type]['fn']
            
            p_type = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r_type = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_type = 2 * p_type * r_type / (p_type + r_type) if (p_type + r_type) > 0 else 0.0
            
            metrics_by_type[ent_type] = {
                'precision': p_type,
                'recall': r_type,
                'f1': f1_type
            }
            
            print(f"  {ent_type}: P={p_type:.3f}, R={r_type:.3f}, F1={f1_type:.3f}")

        # Construir dict completo de métricas
        metrics = {
            'overall': {'precision': p, 'recall': r, 'f1': f1},
            'PER': metrics_by_type['PER'],
            'LOC': metrics_by_type['LOC'],
            'ORG': metrics_by_type['ORG'],
            'counts': {'correct': correct, 'pred_entities': pred_entities_num, 'gold_entities': gold_entities_num}
        }

        # Log metrics if logger available
        if logger and not is_test:
            logger.log_metrics(epoch=epoch, split='dev', metrics=metrics)
        elif logger and is_test:
            logger.log_metrics(epoch=epoch, split='test', metrics=metrics)

        return metrics['overall']['f1'], metrics

    # Training phase.
    print("Start training.")
    instances = read_dataset(args.train_path)

    input_ids = torch.LongTensor([ins[0] for ins in instances])
    label_ids = torch.LongTensor([ins[1] for ins in instances])
    mask_ids = torch.LongTensor([ins[2] for ins in instances])
    pos_ids = torch.LongTensor([ins[3] for ins in instances])
    vm_ids = torch.BoolTensor([ins[4] for ins in instances])
    tag_ids = torch.LongTensor([ins[5] for ins in instances])

    instances_num = input_ids.size(0)
    batch_size = args.batch_size
    train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup, t_total=train_steps)

    total_loss = 0.
    f1 = 0.0
    best_f1 = 0.0

    for epoch in range(1, args.epochs_num+1):
        model.train()
        for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch, tag_ids_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vm_ids, tag_ids)):
            model.zero_grad()

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            tag_ids_batch = tag_ids_batch.to(device)
            vm_ids_batch = vm_ids_batch.long().to(device)

            loss, _, _, _ = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch)
            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i+1, total_loss / args.report_steps))
                if logger:
                    logger.log_loss(epoch, i+1, total_loss / args.report_steps)
                total_loss = 0.

            loss.backward()
            optimizer.step()

        # Evaluation phase.
        print("Start evaluate on dev dataset.")
        f1, dev_metrics = evaluate(args, False, epoch)
        print("Start evaluation on test dataset.")
        evaluate(args, True, epoch)

        if f1 >= best_f1:
            best_f1 = f1
            # save_model(model, args.output_model_path)  # DESHABILITADO: ahorro de espacio
            pass  # Placeholder para evitar IndentationError

    # Evaluation phase.
    print("Final evaluation on test dataset.")

    if torch.cuda.device_count() > 1:
        pass  # model.module.load_state_dict deshabilitado
    else:
        if os.path.exists(args.output_model_path):
            pass  # model.load_state_dict deshabilitado
            print("Loaded best model from {}".format(args.output_model_path))
        else:
            print("Warning: No saved model found at {}. Using current model.".format(args.output_model_path))
    evaluate(args, True, args.epochs_num)


if __name__ == "__main__":
    main()
