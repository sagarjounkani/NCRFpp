# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-02-13 11:49:38

from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from NCRFpp.model.wordsequence import WordSequence
from NCRFpp.model.crf import CRF
from scriptedSeq2Seq.handler_forScript import ModelHandler
from typing import List


class SeqLabel(nn.Module):
    def __init__(self, data):
        super(SeqLabel, self).__init__()
        self.use_crf = data.use_crf
        print("build sequence labeling network...")
        print("use_char: ", data.use_char)

        # original code
        # if data.use_char:
        #     print("char feature extractor: ", data.char_feature_extractor)

        print("char feature extractor: ", data.char_feature_extractor)

        print("word feature extractor: ", data.word_feature_extractor)
        print("use crf: ", self.use_crf)

        self.gpu = data.HP_gpu
        self.average_batch = data.average_batch_loss
        ## add two more label for downlayer lstm, use original label size for CRF
        label_size = data.label_alphabet_size
        data.label_alphabet_size += 2
        self.word_hidden = WordSequence(data)

        # original code
        # if self.use_crf:
        #     self.crf = CRF(label_size, self.gpu)

        # adding code for torch.jit.script compatibility
        self.crf = CRF(label_size, self.gpu)

        # adding attributes for scripting
        self.data_instance2index = {'label_alphabet': data.label_alphabet.instance2index,
                                    'word_alphabet': data.word_alphabet.instance2index,
                                    'char_alphabet': data.char_alphabet.instance2index}
        self.data_instances = {'label_alphabet': data.label_alphabet.instances,
                               'word_alphabet': data.word_alphabet.instances,
                               'char_alphabet': data.char_alphabet.instances}

    # @torch.jit.script_method
    def calculate_loss(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                       char_seq_recover, batch_label, mask):
        outs = self.word_hidden(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                                char_seq_recover)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)

        # if self.use_crf:
        #     total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
        #     scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        # else:
        #     loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
        #     outs = outs.view(batch_size * seq_len, -1)
        #     score = F.log_softmax(outs, 1)
        #     total_loss = loss_function(score, batch_label.view(batch_size * seq_len))
        #     _, tag_seq = torch.max(score, 1)
        #     tag_seq = tag_seq.view(batch_size, seq_len)

        total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
        scores, tag_seq = self.crf._viterbi_decode(outs, mask)

        if self.average_batch:
            total_loss = total_loss / batch_size
        return total_loss, tag_seq

    # @torch.jit.script_method
    def forward(self, word_inputs, feature_inputs: List[torch.Tensor], word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,
                mask):
        outs = self.word_hidden(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                                char_seq_recover)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        if self.use_crf:
            scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        else:
            outs = outs.view(batch_size * seq_len, -1)
            _, tag_seq = torch.max(outs, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
            ## filter padded position with zero
            tag_seq = mask.long() * tag_seq
        return tag_seq

    # def get_lstm_features(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
    #     return self.word_hidden(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)

    # @torch.jit.script_method
    def decode_nbest(self, word_inputs, feature_inputs: List[torch.Tensor], word_seq_lengths, char_inputs, char_seq_lengths,
                     char_seq_recover, mask, nbest: int):
        # if not self.use_crf:
        #     print("Nbest output is currently supported only for CRF! Exit...")
        #     exit(0)
        outs = self.word_hidden(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                                char_seq_recover)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        scores, tag_seq = self.crf._viterbi_decode_nbest(outs, mask, nbest)
        return scores, tag_seq

    @torch.jit.export
    def infer(self, text: str):
        handler = ModelHandler(10, self.data_instance2index, self.data_instances, self.gpu)
        batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, \
        batch_charrecover, mask, batch_wordrecover = handler.preprocess(text)
        # inference
        scores, nbest_tag_seq = self.decode_nbest(batch_word, batch_features, batch_wordlen, batch_char,
                                                  batch_charlen, batch_charrecover, mask, 10)
        return handler.postprocess(scores, nbest_tag_seq, mask, batch_wordrecover, text)
