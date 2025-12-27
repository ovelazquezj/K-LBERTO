# coding: utf-8
"""
KnowledgeGraph - Adapted for Spanish word-level tokenization
Original: Character-level for Chinese
Modified: Word-level for Spanish/European languages
"""
import os
import brain.config as config
import pkuseg
import numpy as np


class KnowledgeGraph(object):
    """
    spo_files - list of Path of *.spo files, or default kg name. e.g., ['HowNet']
    word_level - Boolean flag: True for Spanish/word-level, False for Chinese/character-level
    """

    def __init__(self, spo_files, predicate=False, word_level=True):
        self.predicate = predicate
        self.word_level = word_level  # NEW: Flag to switch between Spanish and Chinese modes
        self.spo_file_paths = [config.KGS.get(f, f) for f in spo_files]
        self.lookup_table = self._create_lookup_table()
        self.segment_vocab = list(self.lookup_table.keys()) + config.NEVER_SPLIT_TAG

        # MODIFIED: Only load Chinese tokenizer if NOT word_level mode
        if not self.word_level:
            # Original: For Chinese character-level tokenization
            self.tokenizer = pkuseg.pkuseg(model_name="default", postag=False, user_dict=self.segment_vocab)
        else:
            # NEW: For Spanish word-level, we don't need pkuseg
            # Words are already pre-tokenized by BETO tokenizer upstream
            self.tokenizer = None

        self.special_tags = set(config.NEVER_SPLIT_TAG)

    def _create_lookup_table(self):
        """
        Create lookup table from SPO (Subject-Predicate-Object) triples.
        
        MODIFIED FOR SPANISH:
        - Ignore predicate completely
        - Use only object values which are real Spanish words in BETO vocab
        - This prevents creating invalid tokens like "sentiment_polaritypositive"
        
        REASON:
        Original K-BERT concatenates pred + obje for Chinese:
          "sentiment_polarity" + "positive" = "sentiment_polaritypositive"
        When BETO tokenizes "sentiment_polaritypositive", it returns [UNK] (ID 100)
        This causes knowledge noise and model collapse.
        
        Solution for Spanish:
          Use only obje: "positive", "negative", "negative_sentiment", etc.
        These are real Spanish words that exist in BETO vocabulary.
        """
        lookup_table = {}
        for spo_path in self.spo_file_paths:
            print("[KnowledgeGraph] Loading spo from {}".format(spo_path))
            with open(spo_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        subj, pred, obje = line.strip().split("\t")
                    except:
                        print("[KnowledgeGraph] Bad spo:", line)
                        continue
                    
                    # CRITICAL FIX: Always use only obje (object), never concatenate with predicate
                    # This is the key difference from original K-BERT (which was designed for Chinese)
                    value = obje  # ✅ ONLY the object value
                    # DO NOT DO: value = pred + obje  # ❌ This creates [UNK] tokens
                    
                    if subj in lookup_table.keys():
                        lookup_table[subj].add(value)
                    else:
                        lookup_table[subj] = set([value])
        return lookup_table

    def add_knowledge_with_vm(self, sent_batch, max_entities=config.MAX_ENTITIES, add_pad=True, max_length=128):
        """
        MODIFIED: Support both Chinese character-level and Spanish word-level tokenization

        input: sent_batch - list of sentences, e.g., ["abcd", "efgh"]
        return: know_sent_batch - list of sentences with entities embedding
                position_batch - list of position index of each token
                visible_matrix_batch - list of visible matrices
                seg_batch - list of segment tags
        """

        # MODIFIED: Conditional tokenization based on word_level flag
        if self.word_level:
            # NEW: Spanish mode - words are already pre-tokenized, just split by spaces
            split_sent_batch = [sent.split() for sent in sent_batch]
        else:
            # Original: Chinese mode - use pkuseg character tokenizer
            split_sent_batch = [self.tokenizer.cut(sent) for sent in sent_batch]

        know_sent_batch = []
        position_batch = []
        visible_matrix_batch = []
        seg_batch = []

        for split_sent in split_sent_batch:

            # create tree - LOGIC SAME FOR BOTH, but meanings differ
            sent_tree = []
            pos_idx_tree = []
            abs_idx_tree = []
            pos_idx = -1
            abs_idx = -1
            abs_idx_src = []

            for token in split_sent:

                entities = list(self.lookup_table.get(token, []))[:max_entities]
                sent_tree.append((token, entities))

                # MODIFIED: Conditional position calculation based on word_level
                if self.word_level:
                    # NEW: For Spanish word-level
                    # Each word = 1 token (not len(word) characters)
                    # This preserves word boundaries for BETO
                    token_pos_idx = [pos_idx + 1]  # One position per word
                    token_abs_idx = [abs_idx + 1]  # One absolute index per word
                else:
                    # Original: For Chinese character-level
                    # Each character is a separate token
                    # len(token) gives number of characters
                    token_pos_idx = [pos_idx+i for i in range(1, len(token)+1)]
                    token_abs_idx = [abs_idx+i for i in range(1, len(token)+1)]

                abs_idx = token_abs_idx[-1]

                # MODIFIED: Entity position calculation also depends on word_level
                entities_pos_idx = []
                entities_abs_idx = []

                for ent in entities:
                    if self.word_level:
                        # NEW: For Spanish, each entity string = 1 token unit
                        # (even if it contains multiple words like "capital of")
                        ent_pos_idx = [token_pos_idx[-1] + 1]  # Next position after token
                        entities_pos_idx.append(ent_pos_idx)
                        ent_abs_idx = [abs_idx + 1]  # Next absolute index
                        abs_idx = ent_abs_idx[-1]
                        entities_abs_idx.append(ent_abs_idx)
                    else:
                        # Original: For Chinese, entity length = number of characters
                        ent_pos_idx = [token_pos_idx[-1] + i for i in range(1, len(ent)+1)]
                        entities_pos_idx.append(ent_pos_idx)
                        ent_abs_idx = [abs_idx + i for i in range(1, len(ent)+1)]
                        abs_idx = ent_abs_idx[-1]
                        entities_abs_idx.append(ent_abs_idx)

                pos_idx_tree.append((token_pos_idx, entities_pos_idx))
                pos_idx = token_pos_idx[-1]
                abs_idx_tree.append((token_abs_idx, entities_abs_idx))
                abs_idx_src += token_abs_idx

            # Get know_sent and pos
            know_sent = []
            pos = []
            seg = []

            for i in range(len(sent_tree)):
                word = sent_tree[i][0]

                if word in self.special_tags:
                    know_sent += [word]
                    seg += [0]
                    pos += pos_idx_tree[i][0]
                else:
                    # MODIFIED: Word-level vs character-level handling
                    if self.word_level:
                        # NEW: For Spanish word-level
                        # Keep word as single unit, don't break into characters
                        know_sent += [word]  # Append word as-is
                        seg += [0]  # One segment tag per word
                        # Position already calculated above (one per word)
                        pos += pos_idx_tree[i][0]
                    else:
                        # Original: For Chinese character-level
                        # Convert word to list of characters
                        add_word = list(word)  # "北京" → ['北', '京']
                        know_sent += add_word
                        seg += [0] * len(add_word)  # One tag per character
                        pos += pos_idx_tree[i][0]

                # Add knowledge entities
                for j in range(len(sent_tree[i][1])):
                    entity = sent_tree[i][1][j]

                    if self.word_level:
                        # NEW: For Spanish word-level
                        # Entity is treated as single token unit
                        know_sent += [entity]  # Append entity as-is
                        seg += [1]  # Mark as knowledge (segment 1)
                        pos += list(pos_idx_tree[i][1][j])
                    else:
                        # Original: For Chinese character-level
                        # Entity converted to list of characters
                        add_word = list(entity)
                        know_sent += add_word
                        seg += [1] * len(add_word)  # Mark all as knowledge
                        pos += list(pos_idx_tree[i][1][j])

            token_num = len(know_sent)

            # Calculate visible matrix
            # MODIFIED: Simplified logic for word-level to avoid character-level complexity
            if self.word_level:
                # NEW: For Spanish word-level
                # Create visible matrix with proper padding mask
                # Tokens can attend to tokens, but NOT to [PAD]
                visible_matrix = np.zeros((max_length, max_length))
                # Allow tokens to attend to each other up to token_num
                visible_matrix[:token_num, :token_num] = 1

                # REASON: Word-level visible matrix is simpler because:
                # 1. No character-level complications
                # 2. Words are meaningful units, not subunits
                # 3. Full visibility prevents knowledge isolation issues
                # 4. Compatible with BETO's word-level attention

            else:
                # Original: For Chinese character-level
                # Complex visible matrix to prevent "knowledge noise"
                visible_matrix = np.zeros((token_num, token_num))
                for item in abs_idx_tree:
                    src_ids = item[0]
                    for id in src_ids:
                        visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
                        visible_matrix[id, visible_abs_idx] = 1
                    for ent in item[1]:
                        for id in ent:
                            visible_abs_idx = ent + src_ids
                            visible_matrix[id, visible_abs_idx] = 1

            src_length = len(know_sent)

            # Padding logic - SAME FOR BOTH MODES
            if len(know_sent) < max_length:
                pad_num = max_length - src_length
                know_sent += [config.PAD_TOKEN] * pad_num
                seg += [0] * pad_num
                pos += [max_length - 1] * pad_num
                # For Chinese: pad visible_matrix. For Spanish: already correct size
                if not self.word_level:
                    visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), 'constant', constant_values=1)
            else:
                know_sent = know_sent[:max_length]
                seg = seg[:max_length]
                pos = pos[:max_length]
                # For Chinese: truncate visible_matrix. For Spanish: already correct size
                if not self.word_level:
                    visible_matrix = visible_matrix[:max_length, :max_length]

            know_sent_batch.append(know_sent)
            position_batch.append(pos)
            visible_matrix_batch.append(visible_matrix)
            seg_batch.append(seg)

        return know_sent_batch, position_batch, visible_matrix_batch, seg_batch
