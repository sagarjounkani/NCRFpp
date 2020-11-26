# -*- coding: utf-8 -*-
# @Author: Max
# @Date:   2018-01-19 11:33:37
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-04-26 13:56:03


"""
Alphabet maps objects to integer ids. It provides two way mapping from the index to the objects.
"""
from __future__ import print_function
import json
import os
import sys
import torch
from typing import Dict, List


@torch.jit.script
class Alphabet:
    def __init__(self, name: str, label: bool=False, keep_growing: bool=True):
        self.name = name
        self.UNKNOWN = "</unk>"
        self.label = label
        self.instance2index = torch.jit.annotate(Dict[str, int], {})
        self.instances = torch.jit.annotate(List[str], [])
        self.keep_growing = keep_growing

        # Index 0 is occupied by default, all else following.
        self.default_index: int = 0
        self.next_index: int = 1
        if not self.label:
            self.add(self.UNKNOWN)

    @torch.jit.unused
    def clear(self, keep_growing=True):
        self.instance2index = {}
        self.instances = []
        self.keep_growing = keep_growing

        # Index 0 is occupied by default, all else following.
        self.default_index = 0
        self.next_index = 1

    def add(self, instance: str):
        if instance not in self.instance2index:
            self.instances.append(instance)
            self.instance2index[instance] = self.next_index
            self.next_index += 1

    def get_index(self, instance: str) -> int:

        # added for scripting
        if instance in self.instance2index:
            return self.instance2index[instance]
        else:
            if self.keep_growing:
                index = self.next_index
                self.add(instance)
                return index
            else:
                return self.instance2index[self.UNKNOWN]

        # commented for scripting
        # try:
        #     return self.instance2index[instance]
        # except KeyError:
        #     if self.keep_growing:
        #         index = self.next_index
        #         self.add(instance)
        #         return index
        #     else:
        #         return self.instance2index[self.UNKNOWN]

    def get_instance(self, index):
        if index == 0:
            if self.label:
                return self.instances[0]
            # First index is occupied by the wildcard element.
            return None

        # added for scripting
        if len(self.instances) >= index > 0:
            return self.instances[index - 1]
        else:
            print('WARNING:Alphabet get_instance ,unknown instance, return the first label.')
            return self.instances[0]

        # commented for scripting
        # try:
        #     return self.instances[index - 1]
        # except IndexError:
        #     print('WARNING:Alphabet get_instance ,unknown instance, return the first label.')
        #     return self.instances[0]

    def size(self):
        # if self.label:
        #     return len(self.instances)
        # else:
        return len(self.instances) + 1

    @torch.jit.unused
    def iteritems(self):
        if sys.version_info[0] < 3:  # If using python3, dict item access uses different syntax
            return self.instance2index.iteritems()
        else:
            return self.instance2index.items()

    @torch.jit.unused
    def enumerate_items(self, start=1):
        if start < 1 or start >= self.size():
            raise IndexError("Enumerate is allowed between [1 : size of the alphabet)")
        return zip(range(start, len(self.instances) + 1), self.instances[start - 1:])

    @torch.jit.unused
    def close(self):
        self.keep_growing = False

    @torch.jit.unused
    def open(self):
        self.keep_growing = True

    @torch.jit.unused
    def get_content(self):
        return {'instance2index': self.instance2index, 'instances': self.instances}

    @torch.jit.unused
    def from_json(self, data):
        self.instances = data["instances"]
        self.instance2index = data["instance2index"]

    @torch.jit.unused
    def save(self, output_directory, name=None):
        """
        Save both alhpabet records to the given directory.
        :param output_directory: Directory to save model and weights.
        :param name: The alphabet saving name, optional.
        :return:
        """
        saving_name = name if name else self.__name
        try:
            json.dump(self.get_content(), open(os.path.join(output_directory, saving_name + ".json"), 'w'))
        except Exception as e:
            print("Exception: Alphabet is not saved: " % repr(e))

    @torch.jit.unused
    def load(self, input_directory, name=None):
        """
        Load model architecture and weights from the give directory. This allow we use old models even the structure
        changes.
        :param input_directory: Directory to save model and weights
        :return:
        """
        loading_name = name if name else self.__name
        self.from_json(json.load(open(os.path.join(input_directory, loading_name + ".json"))))
