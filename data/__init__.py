
from .IndexAnno import AnnoIndexedDataset
from .IndexSrc import SrcIndexedDataset

data_registry={
                 'annoindexed':AnnoIndexedDataset,
                 'srcindexed':SrcIndexedDataset,
         
                 }
