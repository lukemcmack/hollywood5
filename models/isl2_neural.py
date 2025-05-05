import numpy as np , pandas as pd
from ISLP import load_data
from ISLP.models import ModelSpec as MS
import torch
from torch import nn
from torch.optim import RMSprop
from torch.utils.data import TensorDataset
from ISLP.torch import (SimpleDataModule ,SimpleModule ,ErrorTracker ,rec_num_workers)
from ISLP.torch.imdb import (load_lookup ,load_tensor ,load_sparse ,load_sequential )
from bow import get_vocab, popular_words,preprocess
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
import torch.multiprocessing
from torchinfo import summary
from bow import preprocess, get_vocab
# Function to create a tensor that has a binary value if the word is in the reviews
def one_hot(sequences, ncol):
    idx, vals = [], []
    for i, s in enumerate(sequences):
        idx.extend({(i,v):1 for v in s}.keys())
    idx = np.array(idx).T
    vals = np.ones(idx.shape[1], dtype=np.float32)
    tens = torch.sparse_coo_tensor(indices=idx,
                                   values=vals,
                                   size=(len(sequences), ncol))
    return tens.coalesce()




def isl_neural(df_hw,year,words):
    df_hw["clean"]=df_hw[f"Review Text"].apply(lambda x: preprocess(x))
    text=' '.join(df_hw[f"Review Text"].astype(str))
    common_words=get_vocab(preprocess(text),words)



    index_map = {item: idx for idx, item in enumerate(common_words, start=1)}


    df_hw['encoded'] = df_hw["clean"].apply(lambda lst: [index_map.get(item, 0) for item in lst])


    train=df_hw[df_hw['Year Nominated']!=year]

    train=train.copy()
    test=df_hw[df_hw['Year Nominated']==year]
    test=test.copy()

    S_train=list(train['encoded'])

    S_test=list(test['encoded'])
    X_train=one_hot(S_train,words)


    X_test=one_hot(S_test,words)
    L_train=train['Won'].astype(np.float32)
    L_test=test['Won'].astype(np.float32)
    lbxd_train=TensorDataset(X_train.to_dense(),torch.tensor(L_train.values))
    lbxd_test=TensorDataset(X_test.to_dense() ,torch.tensor(L_test.values))

    class LBXD(nn.Module):

        def __init__ (self , input_size ):
            super(LBXD,self).__init__()
            self.dense1 = nn.Linear(input_size , 16)
            self. activation = nn.ReLU()
            self.dense2 = nn.Linear(16, 16)
            self.output = nn.Linear(16, 1)
        
        def forward(self , x):
            val = x
            for _map in [self.dense1 ,self.activation ,self.dense2 ,self.activation ,self.output ]:
                val = _map(val)
            return torch.flatten(val)


    max_num_workers=3
    lbxd_dm=SimpleDataModule(lbxd_train ,lbxd_test,num_workers =min(0,max_num_workers),batch_size =64)
    lbxd_model = LBXD(lbxd_test.tensors[0].size()[1])
    summary(lbxd_model ,input_size = lbxd_test.tensors[0].size(), col_names =['input_size','output_size','num_params'])

    lbxd_optimizer = RMSprop(lbxd_model.parameters(),lr =0.002)
    lbxd_module=SimpleModule.binary_classification(lbxd_model,optimizer=lbxd_optimizer)

    lbxd_trainer = Trainer( deterministic =True ,max_epochs =30 ,callbacks =[ ErrorTracker ()])
    lbxd_trainer.fit(lbxd_module , datamodule =lbxd_dm)

    test_results=lbxd_trainer.test(lbxd_module , datamodule =lbxd_dm)
    predictions = lbxd_trainer.predict(lbxd_module, datamodule=lbxd_dm)

    
    return predictions


def yearly_neural(df,words):
    year_dict = {} 
    for year in df['Year Nominated'].unique():
        predictions=isl_neural(df,year,words)
        print(predictions)
        a=predictions[0][0]
        b=predictions[0][1]
        probs = torch.sigmoid(a)
        is_max = probs == torch.max(probs)
        is_max_numeric = is_max.int() 
        hit=torch.equal(is_max_numeric, b)
        year_dict[year] =hit
        print("For ",year,", The prediction is", hit)
    return pd.DataFrame(year_dict) 


        


















