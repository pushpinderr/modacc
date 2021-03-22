import sys
import json
import torch # uncomment
import argparse 
import numpy as np
import transformers # uncomment
import torch.nn as nn # uncomment
from transformers import AutoModel, AutoTokenizer # uncomment


class LayerNotDefined(Exception):
    '''
    Exception raised when no layer by the supplied name doesn't exist.
    '''
    def __init__(self, layer_name, model_name):
        self.layer_name = layer_name 
        self.model_name = model_name

    def __str__(self):
        return f"The layer \x1b[31;1m{self.layer_name}\x1b[0m doesn't exist for the model, \x1b[31;1m{self.model_name}\x1b[0m!"

    def __repr__(self):
        return self.__str__()


class LayerGetter():
    '''
    LayerGetter class is used to query the model and dump weights to 
    files as needed.
    '''
    def __init__(self, path='bert-base-uncased'):
        '''
        Attributes:
        1. model: the actual transformer model object.
        2. model_path: the path of the model in the hugging face repo.
        2. config: the model configuration metadata.
        3. layer_names: a list of names of all the layers.
        4. layers: a dictionary mapping each tensor to the layer name.
        5. weights: list of the actual tensors for the model weights.
        6. param_count: number of parameters in the model.
        '''
        self.model = AutoModel.from_pretrained(path) #1
        self.model_path = path #2
        self.config = self.model.config #3
        self.layer_names = list(self.model.state_dict().keys()) #4
        self.layers = self.model.state_dict() #5
        self.weights = list(self.layers.values()) #6
        self.param_count = 0 #7
        # update the param_count
        for tensor in self.weights:
            self.param_count += tensor.shape.numel()

    def get_layer_weights(self, layer_name):
        '''
        Arguments:
        1. layer_name: name of layer to be fetched.
        '''
        if layer_name not in self.layer_names:
            raise(LayerNotDefined(layer_name=layer_name, model_name=self.model_path))
        else:
            return self.layers[layer_name]

    def dump_layer_weights(self, layer_name, path=None):
        '''
        Arguments:
        1. layer_name: name of layer to be fetched.
        2. path where it's to be dumped, if not path is supplied, 
           layer_name.json is created and data is dumped there.
        '''
        layer_weights = self.get_layer_weights(layer_name)
        if path:
            pass
        else:
            path = f"{layer_name}.json"
        json.dump(layer_weights.tolist(), open(path, "w"))

    def dump_layer_names(self, path=None):
        '''
        dump all layer names to <path>.txt.
        Arguments:
        1. path: the location to which the txt file having all layer names
           needs to be dumped.
        '''
        if path:
            pass 
        else:
            path = "all_layer_names.txt"
        f = open(path, "w")
        for layer_name in self.layer_names:
            f.write(layer_name + "\n")
        f.close()

    def __str__(self):
        return self.model.__str__()

    def __repr__(self):
        return f"model:{self.model_path}, with {len(self.layers)} layers and {self.param_count} parameters!"

    def get_param_count(self):
        '''
        getter function for the param_count attribute.
        '''
        return self.param_count

    def dump_all(self, path=None):
        '''
        dumo all the weights of the model (the self.layers dictionary), to path.json.
        Arguments:
        1. path: the location to which the json file with the layers is dumped.
        '''
        if path:
            pass
        else:
            path = f"{self.model_path}.json"
        json.dump(self.layers, open(path, "w"))
        
    def print_layer_info(self, layer_name):
        '''
        print info about layer, named "layer_name".
        Arguments:
        1. layer_name: name of the layer whose info is to be printed 
           (name of layer, shape of layer, volume/number of parameters). 
        '''
        shape = self.get_layer_weights(layer_name).shape
        print(f"{layer_name}: shape={shape} volume={shape.numel()}")

    def print_layer_weights(self, layer_name):
        '''
        print weights of a layer, named "layer_name".
        Arguments:
        1. layer_name: name of the layer whose weights are to be printed.
        '''
        print(self.get_layer_weights(layer_name))

    def print(self):
        '''
        print info about all the model weights.
        '''
        for layer_name in self.layer_names:
            self.print_layer_info(layer_name)

    def print_weights(self):
        '''
        print all of the tensors for the model.
        '''
        for layer_name in self.layer_names:
            self.print_layer_weights(layer_name)


def var_compare(x, y, hard=True, thresh=1e-3):
    '''
    A simple implementation of threshold based soft comparison 
    or hard comparison.
    Arguments:
    1. x: first object to be compared
    2. y: second object to be compared
    3. hard: if True do an exact comparison, else do a flexible comparison 
    4. thresh: the tolerance while doing a soft comparison. i.e. abs(x-y) <= thresh
    '''
    if hard:
        return x==y
    else:
        return abs(x-y)<=thresh

def compare_tensors(x, y, hard=True, thresh=1e-3, ret_type='bool'):
    '''
    Compare two tensors, with same shape. Output can be a tensor or
    numpy array for elementwise comaprison, or a single boolean value,
    to check if the two tensors are exactly equal. 
    Arguments:
    1. x: first object to be compared
    2. y: second object to be compared
    3. hard: if True do an exact comparison, else do a flexible comparison 
    4. thresh: the tolerance while doing a soft comparison. i.e. abs(x-y) <= thresh
    5. ret_type: The desired output format is specified using this argument. 
        This argument should assume a value out of ['bool', 'list', 'np', 'pt'] 
        for a boolean, list numpy or pytorch output respectively.
    '''
    assert x.shape == y.shape, "Shapes of tensor don't match"
    if ret_type == 'pt':
        return var_compare(x, y, hard, thresh)
    elif ret_type == 'np':
        return var_compare(x, y, hard, thresh).numpy()
    elif ret_type == 'bool':
        return bool(torch.prod(var_compare(x, y, hard, thresh)))
    elif ret_type == 'list':
        return var_compare(x, y, hard, thresh).tolist()
    else:
        raise(TypeError)


if __name__ == "__main__":
    '''
    Roles played by all the terminal arguments:
    1. the model path
    2. the query (basically a function call)
    3. The remaining arguments are passed to the argument parser, and are the arguments to the function being called
    '''
    assert len(sys.argv)>=3, "need to supply model name and query to layer getter"
    getterObject = LayerGetter(sys.argv[1])
    method_list = [func for func in dir(LayerGetter) if callable(getattr(LayerGetter, func))]
    
    assert sys.argv[2] in method_list, f"the query should belong to {method_list}"
    print(" ".join(sys.argv[3:]))
    parser = argparse.ArgumentParser("get function arguments, using this parser")
    parser.add_argument("-p", "--path")
    parser.add_argument("-n", "--layer_name")
    args = {k:v for k,v in vars(parser.parse_args(sys.argv[3:])).items() if v}
    # print(args)
    getattr(getterObject, sys.argv[2])(**args)