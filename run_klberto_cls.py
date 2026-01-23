# -*- encoding:utf-8 -*-
"""
K-BERT Classification Training for Spanish (TASS Sentiment Analysis)
MODIFIED: Added word_level=True to KnowledgeGraph for Spanish word-level tokenization
Compatible with BETO model and TASS sentiment knowledge graph
Supports visible matrix for knowledge graph injection

CRITICAL FIX (Diciembre 27, 2025):
Fixed segment embedding issue for single-sentence classification (len(line)==2)
PROBLEM: mask was [0,1,1,1...] telling model real tokens are in "segment 1"
         But single-sentence should use segment 0 for all real tokens
SOLUTION: mask now [0,0,0...] for all tokens (consistent single segment)

CRITICAL FIX #2 (Diciembre 27, 2025 - CLASS IMBALANCE):
PROBLEM: NLLLoss() sin pesos → modelo predice siempre clase mayoritaria (0)
REASON: Clase 0 = 41.6% dataset. Predecir siempre clase 0 = 41.6% accuracy
        Intentar discriminar = loss más alto. Modelo elige no aprender.
SOLUTION: NLLLoss(weight=class_weights) con weights inversamente proporcionales
         Clase mayoritaria = peso menor, clases menores = peso mayor
         Ahora discriminar tiene mejor loss que colapsar

PATCHES v7 (Diciembre 28, 2025):
FIX #1: Quitar view() redundante en loss calculation (línea 140)
FIX #2: Agregar dropout en output layers para reducir overfitting
FIX #3: Simplificar máscara (line 220) de confusa a clara
FIX #4: Agregar verificación de colapso a clase mayoritaria en evaluate()
FIX #5: Aumentar epochs a 10 (se ejecuta via script)
"""
import sys
import torch
import json
import random
import argparse
import collections
import torch.nn as nn
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils.tokenizer import *
from uer.model_builder import build_model
from uer.utils.optimizers import  BertAdam
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from brain import KnowledgeGraph
from multiprocessing import Process, Pool
import numpy as np


class BertClassifier(nn.Module):
    def __init__(self, args, model, class_weights=None):
        super(BertClassifier, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        
        # FIX #2: Agregar dropout
        self.dropout = nn.Dropout(args.dropout)
        
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, args.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)

        # CRITICAL FIX #2: Usar class_weights para balancear clases
        if class_weights is not None:
            self.criterion = nn.NLLLoss(weight=class_weights)
            print(f"[BertClassifier] NLLLoss with class weights: {class_weights}")
        else:
            self.criterion = nn.NLLLoss()
            print("[BertClassifier] NLLLoss sin weights (AVISO: puede colapsar a clase mayoritaria)")

        self.use_vm = False if args.no_vm else True
        print("[BertClassifier] use visible_matrix: {}".format(self.use_vm))

    def forward(self, src, label, mask, pos=None, vm=None):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size]
            mask: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, mask, pos)

         # DEBUG: Verificar si embeddings son diferentes
        if label is not None and label.size(0) > 1:
            print(f"\n[DEBUG EMBEDDINGS]")
            print(f"  emb[0, 0, :5]: {emb[0, 0, :5]}")
            print(f"  emb[1, 0, :5]: {emb[1, 0, :5]}")
            print(f"  ¿Idénticos? {torch.allclose(emb[0], emb[1], atol=1e-4)}")


        # Encoder.
        if not self.use_vm:
            vm = None
        output = self.encoder(emb, mask, vm)

         # DEBUG: Verificar si encoder output es diferente
        if label is not None and label.size(0) > 1:
            print(f"[DEBUG ENCODER OUTPUT]")
            print(f"  output[0, 0, :5]: {output[0, 0, :5]}")
            print(f"  output[1, 0, :5]: {output[1, 0, :5]}")
            print(f"  ¿Idénticos? {torch.allclose(output[0], output[1], atol=1e-4)}")


        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]

        # DEBUG: Después del pooling
        if label is not None and output.size(0) > 1:
            print(f"[DEBUG AFTER POOLING]")
            print(f"  output[0]: {output[0, :5]}")
            print(f"  output[1]: {output[1, :5]}")
            print(f"  ¿Idénticos? {torch.allclose(output[0], output[1], atol=1e-4)}")


        output = torch.tanh(self.output_layer_1(output))
        
        # FIX #2: Agregar dropout después del tanh
        output = self.dropout(output)

        # DEBUG: Antes de output_layer_2
        if label is not None and output.size(0) > 1:
            print(f"[DEBUG PRE-OUTPUT_LAYER_2]")
            print(f"  output_layer_2.weight shape: {self.output_layer_2.weight.shape}")
            print(f"  output_layer_2.weight[0, :5]: {self.output_layer_2.weight[0, :5]}")
            print(f"  output_layer_2.bias: {self.output_layer_2.bias}")
            print(f"  input[0, :5]: {output[0, :5]}")
            print(f"  input[1, :5]: {output[1, :5]}\n")


        logits = self.output_layer_2(output)

        # DEBUG: Después de output_layer_2
        if label is not None and logits.size(0) > 1:
            print(f"[DEBUG POST-OUTPUT_LAYER_2]")
            print(f"  logits[0]: {logits[0]}")
            print(f"  logits[1]: {logits[1]}\n")

        # FIX #1: Quitar view() redundante - logits ya tiene forma correcta
        loss = self.criterion(self.softmax(logits), label)
        return loss, logits


def add_knowledge_worker(params):

    p_id, sentences, columns, kg, vocab, args = params

    sentences_num = len(sentences)
    dataset = []
    for line_id, line in enumerate(sentences):
        if line_id % 10000 == 0:
            print("Progress of process {}: {}/{}".format(p_id, line_id, sentences_num))
            sys.stdout.flush()
        line = line.strip().split('\t')
        try:
            if len(line) == 2:
                label = int(line[columns["label"]])
                text = CLS_TOKEN + line[columns["text_a"]]

                tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length)
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")

                token_ids = [vocab.get(t) for t in tokens]
                # ========== CRITICAL FIX ==========
                # BEFORE (WRONG):
                #   mask = [1 if t != PAD_TOKEN else 0 for t in tokens]
                #   Created: [0, 1, 1, 1, 1, ..., 0]
                #   Problem: All real tokens had segment=1, PAD had segment=0
                #
                # WHY IT WAS WRONG:
                #   In embeddings.py: seg_emb = self.segment_embedding(seg)
                #   segment_embedding has size (3, emb_size) → accepts indices [0, 1, 2]
                #   But the logic was: PAD→0, real_tokens→1, additional_segments→2,3,...
                #   For BERT single-sentence: ALL tokens should be SAME segment
                #   Having PAD as different segment than real tokens confused the model
                #
                # AFTER (CORRECT):
                #   All tokens in segment 0 (including PAD)
                #   This is correct for single-sentence classification
                # =================================
                
                # FIX #3: Simplificar máscara - clarificar intención
                # Todos los tokens en segment 0 (single-sentence classification)
                mask = [0 for t in tokens]

                dataset.append((token_ids, label, mask, pos, vm))

            elif len(line) == 3:
                label = int(line[columns["label"]])
                text = CLS_TOKEN + line[columns["text_a"]] + SEP_TOKEN + line[columns["text_b"]] + SEP_TOKEN

                tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length)
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")

                token_ids = [vocab.get(t) for t in tokens]
                mask = []
                seg_tag = 1
                for t in tokens:
                    if t == PAD_TOKEN:
                        mask.append(0)
                    else:
                        mask.append(seg_tag)
                    if t == SEP_TOKEN:
                        seg_tag += 1

                dataset.append((token_ids, label, mask, pos, vm))

            elif len(line) == 4:  # for dbqa
                qid=int(line[columns["qid"]])
                label = int(line[columns["label"]])
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                text = CLS_TOKEN + text_a + SEP_TOKEN + text_b + SEP_TOKEN

                tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length)
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")

                token_ids = [vocab.get(t) for t in tokens]
                mask = []
                seg_tag = 1
                for t in tokens:
                    if t == PAD_TOKEN:
                        mask.append(0)
                    else:
                        mask.append(seg_tag)
                    if t == SEP_TOKEN:
                        seg_tag += 1

                dataset.append((token_ids, label, mask, pos, vm, qid))
            else:
                pass

        except:
            print("Error line: ", line)
    return dataset


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="./models/classifier_model.bin", type=str,
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
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                                   "cnn", "gatedcnn", "attn", \
                                                   "rcnn", "crnn", "gpt", "bilstm"], \
                                                   default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "word", "space"], default="bert",
                        help="Specify the tokenizer. "
                             "BETO uses bert tokenizer for Spanish text. "
                             "Char tokenizer segments sentences into characters. "
                             "Word tokenizer supports online word segmentation based on jieba segmentor. "
                             "Space tokenizer segments sentences into words according to space."
                             )

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=5,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    # Evaluation options.
    parser.add_argument("--mean_reciprocal_rank", action="store_true", help="Evaluation metrics for DBQA dataset.")

    # kg
    parser.add_argument("--kg_name", required=True, help="KG name or path")
    parser.add_argument("--workers_num", type=int, default=1, help="number of process for loading dataset")
    parser.add_argument("--no_vm", action="store_true", help="Disable the visible_matrix")

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Count the number of labels.
    labels_set = set()
    columns = {}
    with open(args.train_path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            try:
                line = line.strip().split("\t")
                if line_id == 0:
                    for i, column_name in enumerate(line):
                        columns[column_name] = i
                    continue
                label = int(line[columns["label"]])
                labels_set.add(label)
            except:
                pass
    args.labels_num = len(labels_set)

    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

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

    # Build knowledge graph (early, needed for training data loading)
    if args.kg_name == 'none':
        spo_files = []
    else:
        spo_files = [args.kg_name]

    kg = KnowledgeGraph(spo_files=spo_files, predicate=True, word_level=True)

    # Dataset loader functions
    def batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms):
        instances_num = input_ids.size()[0]
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i*batch_size: (i+1)*batch_size, :]
            label_ids_batch = label_ids[i*batch_size: (i+1)*batch_size]
            mask_ids_batch = mask_ids[i*batch_size: (i+1)*batch_size, :]
            pos_ids_batch = pos_ids[i*batch_size: (i+1)*batch_size, :]
            vms_batch = vms[i*batch_size: (i+1)*batch_size]
            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch
        if instances_num > instances_num // batch_size * batch_size:
            input_ids_batch = input_ids[instances_num//batch_size*batch_size:, :]
            label_ids_batch = label_ids[instances_num//batch_size*batch_size:]
            mask_ids_batch = mask_ids[instances_num//batch_size*batch_size:, :]
            pos_ids_batch = pos_ids[instances_num//batch_size*batch_size:, :]
            vms_batch = vms[instances_num//batch_size*batch_size:]

            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch

    def read_dataset(path, workers_num=1):

        print("Loading sentences from {}".format(path))
        sentences = []
        with open(path, mode='r', encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                if line_id == 0:
                    continue
                sentences.append(line)
        sentence_num = len(sentences)

        print("There are {} sentence in total. We use {} processes to inject knowledge into sentences.".format(sentence_num, workers_num))
        if workers_num > 1:
            params = []
            sentence_per_block = int(sentence_num / workers_num) + 1
            for i in range(workers_num):
                params.append((i, sentences[i*sentence_per_block: (i+1)*sentence_per_block], columns, kg, vocab, args))
            pool = Pool(workers_num)
            res = pool.map(add_knowledge_worker, params)
            pool.close()
            pool.join()
            dataset = [sample for block in res for sample in block]
        else:
            params = (0, sentences, columns, kg, vocab, args)
            dataset = add_knowledge_worker(params)

        return dataset

    # Evaluation function.
    def evaluate(args, is_test, metrics='Acc', dataset=None):
        if dataset is None:
            if is_test:
                dataset = read_dataset(args.test_path, workers_num=args.workers_num)
            else:
                dataset = read_dataset(args.dev_path, workers_num=args.workers_num)

        input_ids = torch.LongTensor([sample[0] for sample in dataset])
        label_ids = torch.LongTensor([sample[1] for sample in dataset])
        mask_ids = torch.LongTensor([sample[2] for sample in dataset])
        pos_ids = torch.LongTensor([example[3] for example in dataset])
        vms = [example[4] for example in dataset]

        batch_size = args.batch_size
        instances_num = input_ids.size()[0]
        if is_test:
            print("The number of evaluation instances: ", instances_num)

        correct = 0
        # Confusion matrix.
        confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

        model.eval()

        if not args.mean_reciprocal_rank:
            for i, (input_ids_batch, label_ids_batch,  mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms)):

                # vms_batch = vms_batch.long()
                vms_batch = torch.LongTensor(vms_batch)

                input_ids_batch = input_ids_batch.to(device)
                label_ids_batch = label_ids_batch.to(device)
                mask_ids_batch = mask_ids_batch.to(device)
                pos_ids_batch = pos_ids_batch.to(device)
                vms_batch = vms_batch.to(device)

                with torch.no_grad():
                    try:
                        loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch)
                    except Exception as e:
                        print(f"Error en batch {i}: {e}")
                        print(input_ids_batch)
                        print(input_ids_batch.size())
                        print(vms_batch)
                        print(vms_batch.size())
                        raise

                logits = nn.Softmax(dim=1)(logits)
                # DEBUG: Print logits y predicciones para primeros ejemplos
                if i == 0:  # Solo primer batch
                    print("\n[DEBUG] PRIMEROS 5 EJEMPLOS DEL BATCH:")
                    for j in range(min(5, logits.size(0))):
                        print(f"  Ejemplo {j}:")
                        print(f"    Label real: {label_ids_batch[j].item()}")
                        print(f"    Logits softmax: {logits[j].cpu().numpy()}")
                        print(f"    Predicción: {torch.argmax(logits[j]).item()}")
                        print(f"    Confianza: {logits[j].max().item():.4f}")
                    print()
                pred = torch.argmax(logits, dim=1)
                gold = label_ids_batch
                for j in range(pred.size()[0]):
                    confusion[pred[j], gold[j]] += 1
                correct += torch.sum(pred == gold).item()

            if is_test:
                print("Confusion matrix:")
                print(confusion)
                print("Report precision, recall, and f1:")

            for i in range(confusion.size()[0]):
                # MODIFIED: Protected division by zero for imbalanced classes
                # REASON: Spanish dataset may have classes with no predictions
                row_sum = confusion[i,:].sum().item()
                col_sum = confusion[:,i].sum().item()

                if row_sum > 0 and col_sum > 0:
                    p = confusion[i,i].item() / row_sum
                    r = confusion[i,i].item() / col_sum
                    f1 = 2*p*r / (p+r) if (p+r) > 0 else 0
                    if i == 1:
                        label_1_f1 = f1
                    print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i,p,r,f1))
                else:
                    print("Label {}: No samples in this class (skipped)".format(i))

            print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct/len(dataset), correct, len(dataset)))
            
            # FIX #4: Agregar verificación de colapso a clase mayoritaria
            print("\n" + "="*80)
            print("VERIFICACIÓN DE COLAPSO A CLASE MAYORITARIA")
            print("="*80)
            
            # Calcular distribución de clases reales en el dataset
            real_class_dist = torch.zeros(args.labels_num)
            for sample in dataset:
                real_class_dist[sample[1]] += 1
            real_class_dist = real_class_dist / len(dataset)
            
            majority_class_id = real_class_dist.argmax().item()
            majority_class_acc = real_class_dist.max().item()
            current_acc = correct / len(dataset)
            
            print(f"Clase mayoritaria en dataset: {majority_class_id}")
            print(f"Frecuencia de clase mayoritaria: {majority_class_acc:.4f}")
            print(f"Accuracy actual del modelo: {current_acc:.4f}")
            print(f"Diferencia: {abs(current_acc - majority_class_acc):.4f}")
            
            if abs(current_acc - majority_class_acc) < 0.05:
                print("\n⚠️  ALERTA: Modelo está COLAPSANDO a clase mayoritaria")
                print(f"   El modelo predice casi igual que predecir siempre clase {majority_class_id}")
                print(f"   Revisa confusion matrix arriba para ver qué sucede")
                print(f"   Posibles causas:")
                print(f"   - Learning rate muy pequeño")
                print(f"   - Logits demasiado pequeños")
                print(f"   - Knowledge graph no se inyecta correctamente")
                print(f"   - Necesita más epochs para entrenar")
            else:
                print("\n✓ Modelo está discriminando entre clases (diferencia > 0.05)")
            
            print("="*80 + "\n")
            
            if metrics == 'Acc':
                return correct/len(dataset)
            elif metrics == 'f1':
                return label_1_f1 if 'label_1_f1' in locals() else 0
            else:
                return correct/len(dataset)
        else:
            for i, (input_ids_batch, label_ids_batch,  mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms)):

                vms_batch = torch.LongTensor(vms_batch)

                input_ids_batch = input_ids_batch.to(device)
                label_ids_batch = label_ids_batch.to(device)
                mask_ids_batch = mask_ids_batch.to(device)
                pos_ids_batch = pos_ids_batch.to(device)
                vms_batch = vms_batch.to(device)

                with torch.no_grad():
                    loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch)
                logits = nn.Softmax(dim=1)(logits)
                if i == 0:
                    logits_all=logits
                if i >= 1:
                    logits_all=torch.cat((logits_all,logits),0)

            order = -1
            gold = []
            for i in range(len(dataset)):
                qid = dataset[i][-1]
                label = dataset[i][1]
                if qid == order:
                    j += 1
                    if label == 1:
                        gold.append((qid,j))
                else:
                    order = qid
                    j = 0
                    if label == 1:
                        gold.append((qid,j))

            label_order = []
            order = -1
            for i in range(len(gold)):
                if gold[i][0] == order:
                    templist.append(gold[i][1])
                elif gold[i][0] != order:
                    order=gold[i][0]
                    if i > 0:
                        label_order.append(templist)
                    templist = []
                    templist.append(gold[i][1])
            label_order.append(templist)

            order = -1
            score_list = []
            for i in range(len(logits_all)):
                score = float(logits_all[i][1])
                qid=int(dataset[i][-1])
                if qid == order:
                    templist.append(score)
                else:
                    order = qid
                    if i > 0:
                        score_list.append(templist)
                    templist = []
                    templist.append(score)
            score_list.append(templist)

            rank = []
            pred = []
            print(len(score_list))
            print(len(label_order))
            for i in range(len(score_list)):
                if len(label_order[i])==1:
                    if label_order[i][0] < len(score_list[i]):
                        true_score = score_list[i][label_order[i][0]]
                        score_list[i].sort(reverse=True)
                        for j in range(len(score_list[i])):
                            if score_list[i][j] == true_score:
                                rank.append(1 / (j + 1))
                    else:
                        rank.append(0)

                else:
                    true_rank = len(score_list[i])
                    for k in range(len(label_order[i])):
                        if label_order[i][k] < len(score_list[i]):
                            true_score = score_list[i][label_order[i][k]]
                            temp = sorted(score_list[i],reverse=True)
                            for j in range(len(temp)):
                                if temp[j] == true_score:
                                    if j < true_rank:
                                        true_rank = j
                    if true_rank < len(score_list[i]):
                        rank.append(1 / (true_rank + 1))
                    else:
                        rank.append(0)
            MRR = sum(rank) / len(rank)
            print("MRR", MRR)
            return MRR

    # Training phase.
    print("Start training.")
    trainset = read_dataset(args.train_path, workers_num=args.workers_num)

    # CRITICAL FIX #2: Calcular class weights para balancear clases
    print("\n" + "="*80)
    print("CALCULATING CLASS WEIGHTS FOR IMBALANCED DATASET")
    print("="*80)
    class_counts = collections.Counter([sample[1] for sample in trainset])
    print(f"\nClass distribution in training set:")
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        pct = 100 * count / len(trainset)
        print(f"  Clase {class_id}: {count} muestras ({pct:.1f}%)")

    total_samples = len(trainset)
    class_weights = torch.tensor(
        [total_samples / (len(class_counts) * class_counts[i]) for i in range(args.labels_num)],
        dtype=torch.float
    )
    print(f"\nClass weights (inverse frequency):")
    for class_id in range(args.labels_num):
        print(f"  Clase {class_id}: {class_weights[class_id]:.4f}")
    print("="*80 + "\n")

    print("Shuffling dataset")
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size

    print("Trans data to tensor.")
    print("input_ids")
    input_ids = torch.LongTensor([example[0] for example in trainset])
    print("label_ids")
    label_ids = torch.LongTensor([example[1] for example in trainset])
    print("mask_ids")
    mask_ids = torch.LongTensor([example[2] for example in trainset])
    print("pos_ids")
    pos_ids = torch.LongTensor([example[3] for example in trainset])
    print("vms")
    vms = [example[4] for example in trainset]

    train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    # Build classification model CON class_weights
    model = BertClassifier(args, model, class_weights=class_weights)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model = model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup, t_total=train_steps)

    total_loss = 0.
    result = 0.0
    best_result = 0.0

    for epoch in range(1, args.epochs_num+1):
        model.train()
        for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms)):

            # DEBUG: Verificar si inputs son diferentes
            if i == 0:
                print("\n[DEBUG INPUTS] Primeros 3 ejemplos del batch:")
                for j in range(min(3, input_ids_batch.size(0))):
                    print(f"  Ejemplo {j}:")
                    print(f"    input_ids: {input_ids_batch[j][:20]}...")  # Primeros 20 tokens
                    print(f"    label: {label_ids_batch[j].item()}")
                    print(f"    pos_ids: {pos_ids_batch[j][:20]}...")

                    # Verificar si son idénticos
                    if j == 0:
                        first_input = input_ids_batch[j]
                    else:
                        if (first_input == input_ids_batch[j]).all():
                            print(f"    ⚠️ IDÉNTICO AL EJEMPLO 0")
                        else:
                            print(f"    ✓ Diferente del ejemplo 0")
                print()

            model.zero_grad()

            vms_batch = torch.LongTensor(vms_batch)

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            vms_batch = vms_batch.to(device)

            loss, _ = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos=pos_ids_batch, vm=vms_batch)
            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i+1, total_loss / args.report_steps))
                sys.stdout.flush()
                total_loss = 0.
            loss.backward()
            optimizer.step()

        print("Start evaluation on dev dataset.")
        result = evaluate(args, False)
        if result > best_result:
            best_result = result
            save_model(model, args.output_model_path)
        else:
            continue

        print("Start evaluation on test dataset.")
        evaluate(args, True)

    # Evaluation phase.
    print("Final evaluation on the test dataset.")

    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(torch.load(args.output_model_path))
    else:
        model.load_state_dict(torch.load(args.output_model_path))
    evaluate(args, True)


if __name__ == "__main__":
    main()
