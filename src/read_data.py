import os
import csv
import io
from typing import Tuple, List, Iterable, Iterator, Union

import pandas as pd
import numpy as np

class DataExtensions:
    TEXT = "txt"
    CSV = "csv"

class Data:
    def __init__(self, file: str):
        self.file, self.ext = os.path.splitext(file)
        
    @staticmethod
    def read_batches(file_ctx: Iterable, batch_size: int = 10000) -> Iterator:
        batch = list()
        for line in file_ctx:
            batch.append(line)
            if len(batch) == batch_size:
                yield batch
                batch = list()

        # Yield the final batch.
        if batch:
            yield batch
        
    def read_txt(self, 
                 batch_size: int = None, 
                 batch_processing: bool = False) -> Union[Iterator, List]:
        
        with open("".join([self.file, self.ext]), "r") as f:
            if batch_processing:
                for batch in self.read_batches(f, batch_size):
                    yield (x.split() for x in batch)
            else:
                yield [x.split() for x in f.read().splitlines()]
            
    def read_csv(self,
                 batch_size: int = None, 
                 batch_processing: bool = False) -> Union[Iterator, List]:
        
        with open("".join([self.file, self.ext]), "r") as f:
            reader = csv.reader(f, delimiter = ",")
            if batch_processing:
                for batch in self.read_batches(f, batch_size):
                    yield (x.split() for x in batch)
            else:
                yield list(reader)        
            
    def read_file(self, 
                  batch_size: int = None, 
                  batch_processing = False) -> Union[Iterator, List]:
        
        if self.ext == DataExtensions.TEXT:
            raw_data = self.read_txt(batch_size, batch_processing)
        else:
            raw_data = self.read_csv(batch_size, batch_processing) 
            
        return raw_data
    
def main():
    file = "analysis_new.csv"
    dataobj = Data(os.path.join(os.path.abspath("../data"), file))
    print(list(dataobj.read_file()))
    # for batch in dataobj.read_file(batch_size = 10, batch_processing = True):
    #     print(list(batch))
            
if __name__ == "__main__":
    main()
    
            
                
                
                