import numpy as np
import torch


class pdpoint2tensor(object):
    def __init__(self,input_fields,output_fields):
        """
        input and output fields are lists if string for names in the dataframe
        """
        self.input_fields = input_fields
        self.output_fields = output_fields
        #pass
    def __call__(self,datapoint):
        # select fields that are used in the model
        # split in target and input (create a list in INIT)
        x = torch.Tensor(datapoint[self.input_fields],requires_grad=True)
        u = torch.Tensor(datapoint[self.output_fields])
        return {"x":x,"u":u}
        # convert to tensor