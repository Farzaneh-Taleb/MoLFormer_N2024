import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from fast_transformers.masking import LengthMask as LM
import deepchem as dc
import ast
# from util_alignment import *

def batch_split(data, batch_size=64):
    i = 0
    while i < len(data):
        yield data[i:min(i+batch_size, len(data))]
        i += batch_size


def embed(model, smiles, tokenizer, batch_size=64):
    # print(len(model.blocks.layers))
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    model.blocks.layers[0].register_forward_hook(get_activation('0'))
    model.blocks.layers[1].register_forward_hook(get_activation('1'))
    model.blocks.layers[2].register_forward_hook(get_activation('2'))
    model.blocks.layers[3].register_forward_hook(get_activation('3'))
    model.blocks.layers[4].register_forward_hook(get_activation('4'))
    model.blocks.layers[5].register_forward_hook(get_activation('5'))
    model.blocks.layers[6].register_forward_hook(get_activation('6'))
    model.blocks.layers[7].register_forward_hook(get_activation('7'))
    model.blocks.layers[8].register_forward_hook(get_activation('8'))
    model.blocks.layers[9].register_forward_hook(get_activation('9'))
    model.blocks.layers[10].register_forward_hook(get_activation('10'))
    model.blocks.layers[11].register_forward_hook(get_activation('11'))
    model.eval()
    embeddings = []
    keys = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    activations_embeddings = [[],[],[],[],[],[],[],[],[],[],[],[]]
    
    for batch in batch_split(smiles, batch_size=batch_size):
        batch_enc = tokenizer.batch_encode_plus(batch, padding=True, add_special_tokens=True)
        idx, mask = torch.tensor(batch_enc['input_ids']), torch.tensor(batch_enc['attention_mask'])
        with torch.no_grad():
            
            token_embeddings = model.blocks(model.tok_emb(torch.as_tensor(idx)), length_mask=LM(mask.sum(-1)))
            
            input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask
            embeddings.append(embedding.detach().cpu())
            
            for i,key in enumerate(keys):
                transformer_output= activation[key]
                input_mask_expanded = mask.unsqueeze(-1).expand(transformer_output.size()).float()
                sum_embeddings = torch.sum(transformer_output * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embedding = sum_embeddings / sum_mask
                activations_embeddings[i].append(embedding.detach().cpu())
    return embeddings, activations_embeddings
def postproce_molembeddings(embeddings,index):
    # molecules_embeddings_penultimate = torch.cat(embeddings)
    # molecules_embeddings_penultimate=torch.cat(molecules_embeddings_penultimate,index)
    # df_molecules_embeddings = pd.DataFrame(molecules_embeddings_penultimate, index=index)
    # df_molecules_embeddings['Combined'] = df_molecules_embeddings.loc[:, '0':'767'].values.tolist()
    # df_molecules_embeddings=df_molecules_embeddings.reset_index()
    # return(df_molecules_embeddings)
    # if type(embeddings) is tuple:
    molecules_embeddings_penultimate = torch.cat(embeddings)
        # print("1")
    # elif type(embeddings) is list:
        # molecules_embeddings_penultimate = torch.Tensor(np.asarray(embeddings))
        # print("2")
        
        
    columns_size= int(molecules_embeddings_penultimate.size()[1])
    # print("sizeeee",molecules_embeddings_penultimate.size())
    if index.ndim>1:
        
        # print("index", index)
        
        molecules_embeddings_penultimate = torch.cat((  torch.from_numpy( index.to_numpy()),molecules_embeddings_penultimate), dim=1)
        # print("mmmm",molecules_embeddings_penultimate[0:4,0:4])
        df_molecules_embeddings = pd.DataFrame(molecules_embeddings_penultimate,columns=['CID','subject']+[str(i) for i in range(columns_size)])
        # print("ddd",df_molecules_embeddings.columns.tolist())
        
        
        df_molecules_embeddings=df_molecules_embeddings.set_index(['CID','subject'])
        df_molecules_embeddings['Combined'] = df_molecules_embeddings.loc[:, '0':str(columns_size-1)].values.tolist()

        
    else:
        # molecules_embeddings_penultimate = torch.cat((torch.from_numpy(index.to_numpy()).unsqueeze(1),molecules_embeddings_penultimate), dim=1)
        df_molecules_embeddings = pd.DataFrame(molecules_embeddings_penultimate,columns=[str(i) for i in range(columns_size)])
        df_molecules_embeddings['CID']=index
        df_molecules_embeddings=df_molecules_embeddings.set_index(['CID'])
        df_molecules_embeddings['Combined'] = df_molecules_embeddings.loc[:, '0':str(columns_size-1)].values.tolist()
    df_molecules_embeddings=df_molecules_embeddings.reset_index()
    return df_molecules_embeddings


def prepare_mols_helper(modeldeepchem,lm,tokenizer,df_mols,mol_type="nonStereoSMILES",index="CID"):
    df_mols_layers=[]
    df_mols_layers_zscored=[]
    
    #inference on molecules
    df_mols_embeddings_original, df_mols_layers_original=embed(lm,df_mols[mol_type], tokenizer, batch_size=64)
    df_mols_embeddings=postproce_molembeddings(df_mols_embeddings_original,df_mols[index])

     #z-score embeddings
    df_mols_embeddings_zscored = zscore_embeddings(df_mols_embeddings,dim=768)

    #linear transformation of embeddings
    df_mols_embeddings_linear = linear_transformation_embeddings(df_mols, df_mols_embeddings, index, modeldeepchem)

    #z-score linear embeddings
    df_mols_embeddings_linear_zscored = zscore_embeddings(df_mols_embeddings_linear,dim=256)

    for df_mols_layer in df_mols_layers_original:
        df_mols_layer=postproce_molembeddings(df_mols_layer,df_mols[index])
        df_mols_layers.append(df_mols_layer)
        
         #z-score embeddings
        df_mols_embeddings_zscored = zscore_embeddings(df_mols_layer)
        df_mols_layers_zscored.append(df_mols_embeddings_zscored)
        
    
    return df_mols_embeddings_original,df_mols_layers_original,df_mols_embeddings,df_mols_embeddings_zscored,df_mols_layers,df_mols_layers_zscored,df_mols_embeddings_linear,df_mols_embeddings_linear_zscored


def linear_transformation_embeddings(df_mols, df_mols_embeddings, index, modeldeepchem):
    df_mols_embeddings_diskdataset = dc.data.DiskDataset.from_numpy(df_mols_embeddings['Combined'].values.tolist())
    df_mols_embeddings_linear = modeldeepchem.predict_embedding(df_mols_embeddings_diskdataset)
    df_mols_embeddings_linear_torch = [torch.from_numpy(x.reshape(1, -1)) for x in df_mols_embeddings_linear]
    df_mols_embeddings_linear = postproce_molembeddings(df_mols_embeddings_linear_torch, df_mols[index])
    return df_mols_embeddings_linear


def zscore_embeddings(df_mols_embeddings,dim=768):
    df_mols_embeddings_zscored = df_mols_embeddings.copy()
    scaled_features = StandardScaler().fit_transform(df_mols_embeddings_zscored.loc[:, '0':str(dim-1)].values.tolist())
    df_mols_embeddings_zscored.loc[:, '0':str(dim-1)] = pd.DataFrame(scaled_features, index=df_mols_embeddings_zscored.index,
                                                                columns=[str(i) for i in range(dim)])
    df_mols_embeddings_zscored['Combined'] = df_mols_embeddings_zscored.loc[:, '0':str(dim-1)].values.tolist()
    return df_mols_embeddings_zscored


def prepare_mols_helper_mixture(modeldeepchem,df_mols_embeddings_original,df_mols,mol_type="nonStereoSMILES",index="CID",last='255'):
    df_mols_layers=[]
    df_mols_layers_zscored=[]
    
 
    
        
    df_mols_embeddings=postproce_molembeddings(df_mols_embeddings_original,df_mols[index])

    
    
    # df_mols_embeddings_diskdataset = dc.data.DiskDataset.from_numpy(df_mols_embeddings['Combined'].values.tolist())
    # df_mols_embeddings_linear=modeldeepchem.predict_embedding(df_mols_embeddings_diskdataset)
    # df_mols_embeddings_linear_torch=[torch.from_numpy(x.reshape(1,-1)) for x in df_mols_embeddings_linear]
    # df_mols_embeddings_linear=postproce_molembeddings(df_mols_embeddings_linear_torch,df_mols[index])
    
    
     #z-score embeddings
    df_mols_embeddings_zscored = df_mols_embeddings.copy()
    scaled_features = StandardScaler().fit_transform(df_mols_embeddings_zscored.loc[:, '0':last].values.tolist())
    df_mols_embeddings_zscored.loc[:, '0':last] = pd.DataFrame(scaled_features, index=df_mols_embeddings_zscored.index, columns=[str(i) for i in range(int(last)+1)])
    df_mols_embeddings_zscored['Combined'] = df_mols_embeddings_zscored.loc[:, '0':last].values.tolist()
    
    
    
    #z-score linear embeddings
    # df_mols_embeddings_linear_zscored = df_mols_embeddings_linear.copy()
    # scaled_features = StandardScaler().fit_transform(df_mols_embeddings_linear_zscored.loc[:, '0':'255'].values.tolist())
    # df_mols_embeddings_linear_zscored.loc[:, '0':'255'] = pd.DataFrame(scaled_features, index=df_mols_embeddings_linear_zscored.index, columns=[str(i) for i in range(256)])
    # df_mols_embeddings_linear_zscored['Combined'] = df_mols_embeddings_linear_zscored.loc[:, '0':'255'].values.tolist()


    

        
    
    # ÷return df_mols_embeddings_original,df_mols_embeddings,df_mols_embeddings_zscored,df_mols_embeddings_linear,df_mols_embeddings_linear_zscored

    return df_mols_embeddings_original,df_mols_embeddings,df_mols_embeddings_zscored,



# def prepare_mols_helper2(modeldeepchem,df_mols_embeddings_original,df_mols_layers_original,df_mols,mol_type="nonStereoSMILES",index="CID"):
#     df_mols_layers=[]
#     df_mols_layers_zscored=[]
    
#     #inference on molecules
#     # df_mols_embeddings_original, df_mols_layers_original=embed(lm,df_mols[mol_type], tokenizer, batch_size=64)
    
    
        
#     df_mols_embeddings=postproce_molembeddings(df_mols_embeddings_original,df_mols[index])
#     # print("columns",df_mols_embeddings.columns)

    
    
#     df_mols_embeddings_diskdataset = dc.data.DiskDataset.from_numpy(df_mols_embeddings['Combined'].values.tolist())
#     df_mols_embeddings_linear=modeldeepchem.predict_embedding(df_mols_embeddings_diskdataset)
#     df_mols_embeddings_linear_torch=[torch.from_numpy(x.reshape(1,-1)) for x in df_mols_embeddings_linear]
#     df_mols_embeddings_linear=postproce_molembeddings(df_mols_embeddings_linear_torch,df_mols[index])
    
    
#      #z-score embeddings
#     df_mols_embeddings_zscored = df_mols_embeddings.copy()
#     scaled_features = StandardScaler().fit_transform(df_mols_embeddings_zscored.loc[:, '0':'767'].values.tolist())
#     df_mols_embeddings_zscored.loc[:, '0':'767'] = pd.DataFrame(scaled_features, index=df_mols_embeddings_zscored.index, columns=[str(i) for i in range(768)])
#     df_mols_embeddings_zscored['Combined'] = df_mols_embeddings_zscored.loc[:, '0':'767'].values.tolist()
    
    
    
#     #z-score linear embeddings
#     df_mols_embeddings_linear_zscored = df_mols_embeddings_linear.copy()
#     scaled_features = StandardScaler().fit_transform(df_mols_embeddings_linear_zscored.loc[:, '0':'255'].values.tolist())
#     df_mols_embeddings_linear_zscored.loc[:, '0':'255'] = pd.DataFrame(scaled_features, index=df_mols_embeddings_linear_zscored.index, columns=[str(i) for i in range(256)])
#     df_mols_embeddings_linear_zscored['Combined'] = df_mols_embeddings_linear_zscored.loc[:, '0':'255'].values.tolist()


    
#     for df_mols_layer in df_mols_layers_original:
#         df_mols_layer=postproce_molembeddings(df_mols_layer,df_mols[index])
#         df_mols_layers.append(df_mols_layer)
#         # print("step2")
        
#          #z-score embeddings
#         df_mols_embeddings_zscored = df_mols_layer.copy()
#         scaled_features = StandardScaler().fit_transform(df_mols_embeddings_zscored.loc[:, '0':'767'].values.tolist())
#         df_mols_embeddings_zscored.loc[:, '0':'767'] = pd.DataFrame(scaled_features, index=df_mols_embeddings_zscored.index, columns=[str(i) for i in range(768)])
#         df_mols_embeddings_zscored['Combined'] = df_mols_embeddings_zscored.loc[:, '0':'767'].values.tolist()
#         df_mols_layers_zscored.append(df_mols_embeddings_zscored)
        
    
#     return df_mols_embeddings_original,df_mols_layers_original,df_mols_embeddings,df_mols_embeddings_zscored,df_mols_layers,df_mols_layers_zscored,df_mols_embeddings_linear,df_mols_embeddings_linear_zscored


def prepare_keller():
    # input_file_keller = '/local_storage/datasets/farzaneh/openpom/data/curated_datasets/curated_keller2016.csv'
    input_file_keller = '/local_storage/datasets/farzaneh/alignment_olfaction_datasets/curated_datasets/alva/curated_keller2016_nona.csv'
    df_keller=pd.read_csv(input_file_keller)
    df_keller=df_keller.replace(-1000.0, np.NaN)
    df_keller=df_keller.dropna(subset=['Acid', 'Ammonia',
       'Bakery', 'Burnt', 'Chemical', 'Cold', 'Decayed', 'Familiarity', 'Fish',
       'Flower', 'Fruit', 'Garlic', 'Grass', 'Intensity', 'Musky',
       'Pleasantness', 'Sour', 'Spices', 'Sweaty', 'Sweet', 'Warm', 'Wood'])
    n_components=5
    print(df_keller.columns)
    
    #Average of ratings per Molecule
    df_keller_mean =df_keller.groupby(['IsomericSMILES','nonStereoSMILES']).mean().reset_index()
    df_keller_mean['Combined'] = df_keller_mean.loc[:, 'Acid':'Wood'].values.tolist()
    
    #Z-score Keller dataset
    df_keller_zscored = df_keller_mean.copy()
    df_keller_zscored=df_keller_zscored.drop('Combined',axis=1)
    scaled_features = StandardScaler().fit_transform(df_keller_zscored.loc[:, 'Acid':'Wood'].values.tolist())
    df_keller_zscored.loc[:, 'Acid':'Wood'] = pd.DataFrame(scaled_features, index=df_keller_zscored.index, columns=df_keller_zscored.columns[5:])
    
    
    # print("df_keller_zscored.columns[5:]",df_keller_zscored.columns[5:])
    
    #Mean over z-score keller
    df_keller_zscored_mean =df_keller_zscored.groupby(['IsomericSMILES','nonStereoSMILES']).mean().reset_index()
    
    #combine columns
    df_keller_zscored['Combined'] = df_keller_zscored.loc[:, 'Acid':'Wood'].values.tolist()
    df_keller_zscored_mean['Combined'] = df_keller_zscored_mean.loc[:, 'Acid':'Wood'].values.tolist()
    
    
    #PCA on z-scored Keller
    df_keller_zscored_cid_combined = df_keller_zscored[['CID', 'Combined']]
    df_keller_zscored_pca=PCA_df(df_keller_zscored_cid_combined,'Combined' )
    
    #PCA on z-scored_mean Keller
    df_keller_zscored_mean_cid_combined = df_keller_zscored_mean[['CID', 'Combined']]
    df_keller_zscored_mean_pca=PCA_df(df_keller_zscored_mean_cid_combined,'Combined',n_components=n_components )
    
    #Mean on z_scored_PCA
    df_keller_zscored_pca_mean=df_keller_zscored_pca.drop('Combined',axis=1)
    df_keller_zscored_pca_mean =df_keller_zscored_pca_mean.groupby(['CID']).mean().reset_index()
    
    # df_mean_reduced_keller_zscored_cid_combined =df_keller_zscored_pca.groupby(['CID']).mean().reset_index()
    df_keller_zscored_pca_mean['Combined']=df_keller_zscored_pca_mean.loc[:, 0:n_components-1].values.tolist()
    df_keller_zscored_pca_mean=df_keller_zscored_pca_mean.drop([0,1,2,3,4],axis=1)
    
    
    return df_keller, df_keller_mean, df_keller_zscored, df_keller_zscored_mean, df_keller_zscored_pca,df_keller_zscored_mean_pca,df_keller_zscored_pca_mean


def prepare_keller_mols(modeldeepchem_gslf,lm,tokenizer):
    df_keller_mols = df_keller.drop_duplicates('CID')
    print(df_keller_mols.columns)
    df_keller_mols_embeddings_original,df_keller_mols_layers_original,df_keller_mols_embeddings,df_keller_mols_embeddings_zscored,df_keller_mols_layers,df_keller_mols_layers_zscored,df_keller_mols_embeddings_linear,df_keller_mols_embeddings_linear_zscored=prepare_mols_helper(modeldeepchem_gslf,lm,tokenizer,df_keller_mols,mol_type="nonStereoSMILES")
    return df_keller_mols,df_keller_mols_embeddings_original,df_keller_mols_layers_original,df_keller_mols_embeddings,df_keller_mols_embeddings_zscored,df_keller_mols_layers,df_keller_mols_layers_zscored,df_keller_mols_embeddings_linear,df_keller_mols_embeddings_linear_zscored

    
    
def prepare_ravia_backup():
    # input_file = '/local_storage/datasets/farzaneh/openpom/data/curated_datasets/curated_ravia2020_behavior_similairity.csv'
#     pd.read_csv('/local_storage/datasets/farzaneh/openpom/data/curated_datasets/curated_ravia2020_alvaa.csv')
    input_file = '/local_storage/datasets/farzaneh/alignment_olfaction_datasets/curated_datasets/alva/ravia_molecules_alva_17Apr.csv'
    df_ravia_original=pd.read_csv(input_file)
    df_ravia=df_ravia_original.copy()
    print(df_ravia.columns)
    # 'Stimulus 1-IsomericSMILES', 'Stimulus 2-IsomericSMILES',
       # 'Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES'
    
    features= ['CID Stimulus 1','CID Stimulus 2','Stimulus 1-IsomericSMILES','Stimulus 2-IsomericSMILES','Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES', 'RatedSimilarity']
    agg_functions={}
    
    
    chemical_features_r=["nCIR",
                         "ZM1", 
                         "GNar", 
                         "S1K", 
                         "piPC08",
                         "MATS1v",
                         "MATS7v",
                         "GATS1v", 
                         "Eig05_AEA(bo)", 
                         "SM02_AEA(bo)",
                         "SM03_AEA(dm)",
                         "SM10_AEA(dm)",
                         "SM13_AEA(dm)",
                          "SpMin3_Bh(v)",
                         "RDF035v",
                         "G1m",
                         "G1v",
                         "G1e",
                         "G3s",
                         "R8u+",
                         "nRCOSR"]

    
    nonStereoSMILE1 = list(map(lambda x: "Stimulus 1-nonStereoSMILES___" + x, chemical_features_r))
    nonStereoSMILE2 = list(map(lambda x: "Stimulus 2-nonStereoSMILES___" + x, chemical_features_r))
    IsomericSMILES1 = list(map(lambda x: "Stimulus 1-IsomericSMILES___" + x, chemical_features_r))
    IsomericSMILES2 = list(map(lambda x: "Stimulus 2-IsomericSMILES___" + x, chemical_features_r))
   
    chemical_features = nonStereoSMILE1+nonStereoSMILE2+IsomericSMILES1+IsomericSMILES2
    keys = chemical_features.copy()
    values = [chemical_aggregator]*len(chemical_features)

    # Create the dictionary using a dictionary comprehension
    agg_functions = {key: value for key, value in zip(keys, values)}        
        
    features_all = features + chemical_features

    df_ravia=df_ravia.reindex(columns=features_all)
        
    agg_functions['RatedSimilarity'] = 'mean'
    # print(agg_functions,"agg_functions")
    # print(features_all)
    
    
    df_ravia = df_ravia[ features_all]
    df_ravia_copy = df_ravia.copy()
    df_ravia_copy = df_ravia_copy.rename(columns={'Stimulus 1-IsomericSMILES': 'Stimulus 2-IsomericSMILES', 'Stimulus 2-IsomericSMILES': 'Stimulus 1-IsomericSMILES', 'CID Stimulus 1': 'CID Stimulus 2', 'CID Stimulus 2': 'CID Stimulus 1','Stimulus 1-nonStereoSMILES': 'Stimulus 2-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES': 'Stimulus 1-nonStereoSMILES'})
    df_ravia_copy['RatedSimilarity']=np.nan
    df_ravia_concatenated= pd.concat([df_ravia, df_ravia_copy], ignore_index=True, axis=0).reset_index(drop=True)
    df_ravia=df_ravia_concatenated.drop_duplicates(['CID Stimulus 1','CID Stimulus 2','Stimulus 1-IsomericSMILES','Stimulus 2-IsomericSMILES','Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES'])

    
    # df_ravia_mean =df_ravia.groupby(['CID Stimulus 1','CID Stimulus 2','Stimulus 1-IsomericSMILES','Stimulus 2-IsomericSMILES','Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES']).mean().reset_index()
    # df_ravia_mean=df_ravia_mean.drop(columns=['CID Stimulus 1','CID Stimulus 2','Stimulus 1-IsomericSMILES','Stimulus 2-IsomericSMILES','Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES'])
    
    
    df_ravia_mean =df_ravia.groupby(['CID Stimulus 1','CID Stimulus 2','Stimulus 1-IsomericSMILES','Stimulus 2-IsomericSMILES','Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES']).agg(agg_functions).reset_index()
    # df_ravia_mean=df_ravia_mean.drop(columns=['CID Stimulus 1','CID Stimulus 2','Stimulus 1-IsomericSMILES','Stimulus 2-IsomericSMILES','Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES'])
    
    # result_df = df_ravia.groupby('category').agg(agg_functions)
    
    df_ravia_mean_pivoted = df_ravia_mean.pivot(index='CID Stimulus 1', columns='CID Stimulus 2', values='RatedSimilarity')
    # df_ravia_mean_pivoted.head(5)
    df_ravia_mean_pivoted = df_ravia_mean_pivoted.reindex(sorted(df_ravia_mean_pivoted.columns), axis=1)
    df_ravia_mean_pivoted=df_ravia_mean_pivoted.sort_index(ascending=True)
    
    
    return  df_ravia_original,df_ravia_mean,df_ravia_mean_pivoted





def prepare_ravia_or_snitz(dataset,base_path='/local_storage/datasets/farzaneh/alignment_olfaction_datasets/'):
    # generate docstrings for this function with a brief description of the function and the parameters and return values
    """
    Prepare the similarity dataset for the alignment task
    :param base_path:   (str) path to the base directory where the datasets are stored
    :return:         (tuple) a tuple containing the original  dataset, the mean  dataset, and the pivoted mean  dataset
    """

    input_file = base_path + dataset
    df_ravia_original = pd.read_csv(input_file)
    df_ravia = df_ravia_original.copy()

    features = ['CID Stimulus 1', 'CID Stimulus 2', 'Stimulus 1-IsomericSMILES', 'Stimulus 2-IsomericSMILES',
                'Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES', 'RatedSimilarity']
    agg_functions = {}
    features_all = features
    df_ravia = df_ravia.reindex(columns=features_all)

    agg_functions['RatedSimilarity'] = 'mean'

    df_ravia = df_ravia[features_all]
    df_ravia_copy = df_ravia.copy()
    df_ravia_copy = df_ravia_copy.rename(columns={'Stimulus 1-IsomericSMILES': 'Stimulus 2-IsomericSMILES',
                                                  'Stimulus 2-IsomericSMILES': 'Stimulus 1-IsomericSMILES',
                                                  'CID Stimulus 1': 'CID Stimulus 2',
                                                  'CID Stimulus 2': 'CID Stimulus 1',
                                                  'Stimulus 1-nonStereoSMILES': 'Stimulus 2-nonStereoSMILES',
                                                  'Stimulus 2-nonStereoSMILES': 'Stimulus 1-nonStereoSMILES'})
    df_ravia_copy['RatedSimilarity'] = np.nan
    df_ravia_concatenated = pd.concat([df_ravia, df_ravia_copy], ignore_index=True, axis=0).reset_index(drop=True)
    df_ravia = df_ravia_concatenated.drop_duplicates(
        ['CID Stimulus 1', 'CID Stimulus 2', 'Stimulus 1-IsomericSMILES', 'Stimulus 2-IsomericSMILES',
         'Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES'])

    df_ravia_mean = df_ravia.groupby(
        ['CID Stimulus 1', 'CID Stimulus 2', 'Stimulus 1-IsomericSMILES', 'Stimulus 2-IsomericSMILES',
         'Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES']).agg(agg_functions).reset_index()

    df_ravia_mean_pivoted = df_ravia_mean.pivot(index='CID Stimulus 1', columns='CID Stimulus 2',
                                                values='RatedSimilarity')

    df_ravia_mean_pivoted = df_ravia_mean_pivoted.reindex(sorted(df_ravia_mean_pivoted.columns), axis=1)
    df_ravia_mean_pivoted = df_ravia_mean_pivoted.sort_index(ascending=True)

    return df_ravia_original, df_ravia_mean, df_ravia_mean_pivoted


def prepare_ravia_sep():

    input_file = '/local_storage/datasets/farzaneh/alignment_olfaction_datasets/curated_datasets/mols_datasets/curated_ravia2020_behavior_similairity.csv'
    df_ravia_original=pd.read_csv(input_file)
    df_ravia=df_ravia_original.copy()
    print(df_ravia.columns)
    # 'Stimulus 1-IsomericSMILES', 'Stimulus 2-IsomericSMILES',
       # 'Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES'
    
    features= ['CID Stimulus 1','CID Stimulus 2','Stimulus 1-IsomericSMILES_sep','Stimulus 2-IsomericSMILES_sep','Stimulus 1-nonStereoSMILES_sep', 'Stimulus 2-nonStereoSMILES_sep', 'RatedSimilarity']
    agg_functions={}
    
    features_all = features
    df_ravia=df_ravia.reindex(columns=features_all)
        
    agg_functions['RatedSimilarity'] = 'mean'
    # print(agg_functions,"agg_functions")
    # print(features_all)
    
    
    df_ravia = df_ravia[ features_all]
    df_ravia_copy = df_ravia.copy()
    df_ravia_copy = df_ravia_copy.rename(columns={'Stimulus 1-IsomericSMILES_sep': 'Stimulus 2-IsomericSMILES_sep', 'Stimulus 2-IsomericSMILES_sep': 'Stimulus 1-IsomericSMILES_sep', 'CID Stimulus 1': 'CID Stimulus 2', 'CID Stimulus 2': 'CID Stimulus 1','Stimulus 1-nonStereoSMILES_sep': 'Stimulus 2-nonStereoSMILES_sep', 'Stimulus 2-nonStereoSMILES_sep': 'Stimulus 1-nonStereoSMILES_sep'})
    df_ravia_copy['RatedSimilarity']=np.nan
    df_ravia_concatenated= pd.concat([df_ravia, df_ravia_copy], ignore_index=True, axis=0).reset_index(drop=True)
    df_ravia=df_ravia_concatenated.drop_duplicates(['CID Stimulus 1','CID Stimulus 2','Stimulus 1-IsomericSMILES_sep','Stimulus 2-IsomericSMILES_sep','Stimulus 1-nonStereoSMILES_sep', 'Stimulus 2-nonStereoSMILES_sep'])

    
    # df_ravia_mean =df_ravia.groupby(['CID Stimulus 1','CID Stimulus 2','Stimulus 1-IsomericSMILES','Stimulus 2-IsomericSMILES','Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES']).mean().reset_index()
    # df_ravia_mean=df_ravia_mean.drop(columns=['CID Stimulus 1','CID Stimulus 2','Stimulus 1-IsomericSMILES','Stimulus 2-IsomericSMILES','Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES'])
    
    
    df_ravia_mean =df_ravia.groupby(['CID Stimulus 1','CID Stimulus 2','Stimulus 1-IsomericSMILES_sep','Stimulus 2-IsomericSMILES_sep','Stimulus 1-nonStereoSMILES_sep', 'Stimulus 2-nonStereoSMILES_sep']).agg(agg_functions).reset_index()
    # df_ravia_mean=df_ravia_mean.drop(columns=['CID Stimulus 1','CID Stimulus 2','Stimulus 1-IsomericSMILES','Stimulus 2-IsomericSMILES','Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES'])
    
    # result_df = df_ravia.groupby('category').agg(agg_functions)
    
    df_ravia_mean_pivoted = df_ravia_mean.pivot(index='CID Stimulus 1', columns='CID Stimulus 2', values='RatedSimilarity')
    # df_ravia_mean_pivoted.head(5)
    df_ravia_mean_pivoted = df_ravia_mean_pivoted.reindex(sorted(df_ravia_mean_pivoted.columns), axis=1)
    df_ravia_mean_pivoted=df_ravia_mean_pivoted.sort_index(ascending=True)
    
    
    return  df_ravia_original,df_ravia_mean,df_ravia_mean_pivoted

def prepare_ravia_similarity_mols2(df_ravia_similarity_mean,modeldeepchem_gslf,lm,tokenizer):
    df_ravia_mean_mols1 = df_ravia_similarity_mean[['Stimulus 1-IsomericSMILES','Stimulus 1-nonStereoSMILES','CID Stimulus 1']].drop_duplicates().reset_index(drop=True)
    df_ravia_mean_mols2 = df_ravia_similarity_mean[['Stimulus 2-IsomericSMILES','Stimulus 2-nonStereoSMILES','CID Stimulus 2']].drop_duplicates().reset_index(drop=True).rename(columns={'Stimulus 2-nonStereoSMILES': 'Stimulus 1-nonStereoSMILES','Stimulus 2-IsomericSMILES':'Stimulus 1-IsomericSMILES', 'CID Stimulus 2': 'CID Stimulus 1' })
    df_ravia_mols= pd.concat([df_ravia_mean_mols1, df_ravia_mean_mols2], ignore_index=True, axis=0).reset_index(drop=True)
    df_ravia_mols=df_ravia_mols.drop_duplicates().reset_index(drop=True)
    df_ravia_mols = df_ravia_mols.rename(columns={'Stimulus 1-IsomericSMILES': 'IsomericSMILES','Stimulus 1-nonStereoSMILES':'nonStereoSMILES', 'CID Stimulus 1': 'CID' })
    mol_type="nonStereoSMILES"


    df_ravia_mols_embeddings_original,df_ravia_mols_layers_original,df_ravia_mols_embeddings,df_ravia_mols_embeddings_zscored,df_ravia_mols_layers,df_ravia_mols_layers_zscored,df_ravia_mols_embeddings_linear,df_ravia_mols_embeddings_linear_zscored=prepare_mols_helper(modeldeepchem_gslf,lm,tokenizer,df_ravia_mols)
    
    return df_ravia_mols,df_ravia_mols_embeddings_original,df_ravia_mols_layers_original,df_ravia_mols_embeddings,df_ravia_mols_embeddings_zscored,df_ravia_mols_layers,df_ravia_mols_layers_zscored,df_ravia_mols_embeddings_linear,df_ravia_mols_embeddings_linear_zscored

def sum_embeddings(cid_list, df_embeddings):
    embedding_sum = np.zeros(len(df_embeddings.iloc[0]['embeddings']))
    for cid in cid_list:
        if cid in df_embeddings['CID'].values:
            embedding_sum += df_embeddings.loc[df_embeddings['CID'] == cid, 'embeddings'].values[0]
    return embedding_sum


def extract_embeddings(cid, df_embeddings):
    embedding_sum = np.zeros(len(df_embeddings.iloc[0]['embeddings']))
    # for cid in cid_list:
    if cid in df_embeddings['CID'].values:
        embedding_sum += df_embeddings.loc[df_embeddings['CID'] == cid, 'embeddings'].values[0]
    return embedding_sum


def average_embeddings(cid_list, df_embeddings):
    embedding_sum = np.zeros(len(df_embeddings.iloc[0]['embeddings']))
    for cid in cid_list:
        if cid in df_embeddings['CID'].values:
            embedding_sum += df_embeddings.loc[df_embeddings['CID'] == cid, 'embeddings'].values[0]
    return embedding_sum



def prepare_ravia_similarity_mols_mixture(input_file_embeddings,df_ravia_similarity_mean,modeldeepchem_gslf):
    df_ravia_mean_mols1 = df_ravia_similarity_mean[['Stimulus 1-IsomericSMILES','Stimulus 1-nonStereoSMILES','CID Stimulus 1']].drop_duplicates().reset_index(drop=True)
    df_ravia_mean_mols2 = df_ravia_similarity_mean[['Stimulus 2-IsomericSMILES','Stimulus 2-nonStereoSMILES','CID Stimulus 2']].drop_duplicates().reset_index(drop=True).rename(columns={'Stimulus 2-nonStereoSMILES': 'Stimulus 1-nonStereoSMILES','Stimulus 2-IsomericSMILES':'Stimulus 1-IsomericSMILES', 'CID Stimulus 2': 'CID Stimulus 1' })
    df_ravia_mols= pd.concat([df_ravia_mean_mols1, df_ravia_mean_mols2], ignore_index=True, axis=0).reset_index(drop=True)
    df_ravia_mols=df_ravia_mols.drop_duplicates().reset_index(drop=True)
    df_ravia_mols = df_ravia_mols.rename(columns={'Stimulus 1-IsomericSMILES': 'IsomericSMILES','Stimulus 1-nonStereoSMILES':'nonStereoSMILES', 'CID Stimulus 1': 'CID' })
    mol_type="nonStereoSMILES"



    # input_file_embeddings = '/local_storage/datasets/farzaneh/alignment_olfaction_datasets/curated_datasets/embeddings/pom/ravia_pom_embeddings_Apr17.csv'
    df_embeddigs = pd.read_csv(input_file_embeddings)[['embeddings','CID']]
    df_embeddigs['embeddings'] = df_embeddigs['embeddings'].apply(lambda x: np.array(eval(x)))


    df_ravia_mols['Stimulus Embedding Sum'] = df_ravia_mols['CID'].apply(lambda x: sum_embeddings(list(map(int, x.split(';'))), df_embeddigs))
    df_mols_embeddings_original =[torch.from_numpy(np.asarray(df_ravia_mols['Stimulus Embedding Sum'].values.tolist()))]


    # df_ravia_mols.to_csv('df_ravia_mols.csv')  



    df_ravia_mols_embeddings_original,df_ravia_mols_embeddings,df_ravia_mols_embeddings_zscored=prepare_mols_helper_mixture(modeldeepchem_gslf,df_mols_embeddings_original,df_ravia_mols)
    
    return df_ravia_mols,df_ravia_mols_embeddings_original,df_ravia_mols_embeddings,df_ravia_mols_embeddings_zscored



def prepare_ravia_similarity_mols(df_ravia_similarity_mean,modeldeepchem_gslf,lm,tokenizer):
    df_ravia_mean_mols1 = df_ravia_similarity_mean[['Stimulus 1-IsomericSMILES','Stimulus 1-nonStereoSMILES','CID Stimulus 1']].drop_duplicates().reset_index(drop=True)
    df_ravia_mean_mols2 = df_ravia_similarity_mean[['Stimulus 2-IsomericSMILES','Stimulus 2-nonStereoSMILES','CID Stimulus 2']].drop_duplicates().reset_index(drop=True).rename(columns={'Stimulus 2-nonStereoSMILES': 'Stimulus 1-nonStereoSMILES','Stimulus 2-IsomericSMILES':'Stimulus 1-IsomericSMILES', 'CID Stimulus 2': 'CID Stimulus 1' })
    df_ravia_mols= pd.concat([df_ravia_mean_mols1, df_ravia_mean_mols2], ignore_index=True, axis=0).reset_index(drop=True)
    df_ravia_mols=df_ravia_mols.drop_duplicates().reset_index(drop=True)
    df_ravia_mols = df_ravia_mols.rename(columns={'Stimulus 1-IsomericSMILES': 'IsomericSMILES','Stimulus 1-nonStereoSMILES':'nonStereoSMILES', 'CID Stimulus 1': 'CID' })
    # mol_type="nonStereoSMILES"
    # df_ravia_mols.to_csv('df_ravia_mols.csv')  

    df_mols_embeddings_original, df_mols_layers_original=embed(lm,df_ravia_mols[mol_type], tokenizer, batch_size=64)



    # input_file_molformer = '/local_storage/datasets/farzaneh/alignment_olfaction_datasets/curated_datasets/embeddings/molformer/ravia_molformer_embeddings_13_Apr17.csv'
    # gs_lf_molformer=pd.read_csv(input_file_molformer)
    # gs_lf_molformer['embeddings'] = gs_lf_molformer['embeddings'].apply(ast.literal_eval)
    # df_mols_embeddings_original = gs_lf_molformer['embeddings'].tolist()
    
    
    # df_mols_layers_original = []

    # for i in range(12):
    #     input_file_molformer = '/local_storage/datasets/farzaneh/alignment_olfaction_datasets/curated_datasets/embeddings/molformer/ravia_molformer_embeddings_'+str(i)+'_Apr17.csv'
    #     gs_lf_molformer=pd.read_csv(input_file_molformer)
    #     gs_lf_molformer['embeddings'] = gs_lf_molformer['embeddings'].apply(ast.literal_eval)
    #     df_mols_embeddings_original = gs_lf_molformer['embeddings'].tolist()
    #     df_mols_layers_original.append(df_mols_embeddings_original)
        
# df_mols_embeddings_original,df_mols_layers_original

    

    df_ravia_mols_embeddings_original,df_ravia_mols_layers_original,df_ravia_mols_embeddings ,df_ravia_mols_embeddings_zscored,df_ravia_mols_layers,df_ravia_mols_layers_zscored,df_ravia_mols_embeddings_linear,df_ravia_mols_embeddings_linear_zscored=prepare_mols_helper(df_mols_embeddings_original,df_mols_layers_original,df_ravia_mols)
    
    return df_ravia_mols,df_ravia_mols_embeddings_original,df_ravia_mols_layers_original,df_ravia_mols_embeddings,df_ravia_mols_embeddings_zscored,df_ravia_mols_layers,df_ravia_mols_layers_zscored,df_ravia_mols_embeddings_linear,df_ravia_mols_embeddings_linear_zscored

def prepare_snitz_backup():
    # input_file = '/local_storage/datasets/farzaneh/openpom/data/curated_datasets/curated_snitz2013.csv'
    input_file = '/local_storage/datasets/farzaneh/alignment_olfaction_datasets/curated_datasets/alva/snitz_molecules_alva_17Apr.csv'
    
    df_snitz=pd.read_csv(input_file, low_memory=False)
    
    
    
    features= ['CID Stimulus 1','CID Stimulus 2','Stimulus 1-IsomericSMILES','Stimulus 2-IsomericSMILES','Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES', 'Similarity']
    agg_functions={}
    
    
    chemical_features_r=["nCIR",
                         "ZM1", 
                         "GNar", 
                         "S1K", 
                         "piPC08",
                         "MATS1v",
                         "MATS7v",
                         "GATS1v", 
                         "Eig05_AEA(bo)", 
                         "SM02_AEA(bo)",
                         "SM03_AEA(dm)",
                         "SM10_AEA(dm)",
                         "SM13_AEA(dm)",
                          "SpMin3_Bh(v)",
                         "RDF035v",
                         "G1m",
                         "G1v",
                         "G1e",
                         "G3s",
                         "R8u+",
                         "nRCOSR"]

    
    nonStereoSMILE1 = list(map(lambda x: "Stimulus 1-nonStereoSMILES___" + x, chemical_features_r))
    nonStereoSMILE2 = list(map(lambda x: "Stimulus 2-nonStereoSMILES___" + x, chemical_features_r))
    IsomericSMILES1 = list(map(lambda x: "Stimulus 1-IsomericSMILES___" + x, chemical_features_r))
    IsomericSMILES2 = list(map(lambda x: "Stimulus 2-IsomericSMILES___" + x, chemical_features_r))
    
    
    
    chemical_features = nonStereoSMILE1+nonStereoSMILE2+IsomericSMILES1+IsomericSMILES2
    keys = chemical_features.copy()
    values = [chemical_aggregator]*len(chemical_features)

    # Create the dictionary using a dictionary comprehension
    agg_functions = {key: value for key, value in zip(keys, values)}        
        
    features_all = features + chemical_features
    
    df_snitz=df_snitz.reindex(columns=features_all)
        
    agg_functions['Similarity'] = 'mean'
    # print(agg_functions,"agg_functions")
    # print(features_all)
    
    
    df_snitz = df_snitz[ features_all]

    
    
    df_snitz_copy = df_snitz.copy()
    df_snitz_copy = df_snitz_copy.rename(columns={'Stimulus 1-IsomericSMILES': 'Stimulus 2-IsomericSMILES', 'Stimulus 2-IsomericSMILES': 'Stimulus 1-IsomericSMILES', 'CID Stimulus 1': 'CID Stimulus 2', 'CID Stimulus 2': 'CID Stimulus 1','Stimulus 1-nonStereoSMILES': 'Stimulus 2-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES': 'Stimulus 1-nonStereoSMILES'})
    df_snitz_copy['RatedSimilarity']=np.nan
    df_snitz_concatenated= pd.concat([df_snitz, df_snitz_copy], ignore_index=True, axis=0).reset_index(drop=True)
    df_snitz=df_snitz_concatenated.drop_duplicates(['CID Stimulus 1','CID Stimulus 2','Stimulus 1-IsomericSMILES','Stimulus 2-IsomericSMILES','Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES'])

    
    df_snitz_mean =df_snitz.groupby(['CID Stimulus 1','CID Stimulus 2','Stimulus 1-IsomericSMILES','Stimulus 2-IsomericSMILES','Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES']).agg(agg_functions).reset_index()
    # df_ravia_mean=df_ravia_mean.drop(columns=['Stimulus 1', 'Stimulus 2'])
    
    df_snitz_mean_pivoted = df_snitz_mean.pivot(index='CID Stimulus 1', columns='CID Stimulus 2', values='Similarity')
    df_snitz_mean_pivoted = df_snitz_mean_pivoted.reindex(sorted(df_snitz_mean_pivoted.columns), axis=1)
    df_snitz_mean_pivoted=df_snitz_mean_pivoted.sort_index(ascending=True)
    
    
    
    return df_snitz, df_snitz_mean,df_snitz_mean_pivoted




def prepare_sagar():
    
    input_file_sagar = '/local_storage/datasets/farzaneh/alignment_olfaction_datasets/curated_datasets/alva/sagar_molecules_alva_17Apr.csv'
    df_sagar=pd.read_csv(input_file_sagar)
    df_sagar = df_sagar.rename(columns={"cid":"CID"})
    
    columns_list = ['Intensity', 'Pleasantness', 'Fishy', 'Burnt', 'Sour', 'Decayed',
       'Musky', 'Fruity', 'Sweaty', 'Cool', 'Chemical', 'Floral', 'Sweet',
       'Warm', 'Bakery', 'Garlic', 'Spicy', 'Acidic', 'Ammonia', 'Edible','Familiar']
    
    
    df_sagar_common=df_sagar.copy()
    
      # Specify your list of columns

    # Find columns with NaN values
    columns_with_nan = df_sagar_common.columns[df_sagar_common.isna().any()].tolist()
    
    # Find columns that are both in the list and contain NaN values
    columns_to_drop = list(set(columns_list) & set(columns_with_nan))
    
    # Drop columns from DataFrame
    df_sagar_common = df_sagar_common.drop(columns=columns_to_drop)
        
    
    
    
    
    # df_sagar = df_sagar.dropna(axis=1)
    
    # df_sagar_mean =df_sagar.groupby(['IsomericSMILES','nonStereoSMILES']).mean().reset_index()
    df_sagar_mean =df_sagar.groupby(['IsomericSMILES','nonStereoSMILES']).apply(lambda x: x.iloc[:, :-4].mean()).reset_index()
    
    
    df_sagar_mean['Combined'] = df_sagar_mean.loc[:, columns_list].values.tolist()
    df_sagar['Combined'] = df_sagar.loc[:, columns_list].values.tolist()
    # return df_sagar_mean
    
#     #Z-score sagar dataset
    df_sagar_zscored = df_sagar_mean.copy()
    df_sagar_zscored=df_sagar_zscored.drop('Combined',axis=1)
    scaled_features = StandardScaler().fit_transform(df_sagar_zscored.loc[:,columns_list].values.tolist())
    df_sagar_zscored.loc[:, columns_list] = pd.DataFrame(scaled_features, index=df_sagar_zscored.index, columns=columns_list)
    
#     #Mean over z-score sagar
    df_sagar_zscored_mean =df_sagar_zscored.groupby(['IsomericSMILES','nonStereoSMILES']).mean().reset_index()
    
    #combine columns
    df_sagar_zscored['Combined'] = df_sagar_zscored.loc[:, columns_list].values.tolist()
    df_sagar_zscored_mean['Combined'] = df_sagar_zscored_mean.loc[:, columns_list].values.tolist()
    
    
    #PCA on z-scored sagar
    df_sagar_zscored_cid_combined = df_sagar_zscored[['CID', 'Combined']]
    # df_sagar_zscored_pca=PCA_df(df_sagar_zscored_cid_combined,'Combined' )
    
    #PCA on z-scored_mean sagar
    df_sagar_zscored_mean_cid_combined = df_sagar_zscored_mean[['CID', 'Combined']]
    # df_sagar_zscored_mean_pca=PCA_df(df_sagar_zscored_mean_cid_combined,'Combined',n_components=n_components )
    
    #Mean on z_scored_PCA
    # df_sagar_zscored_pca_mean=df_sagar_zscored_pca.drop('Combined',axis=1)
    # df_sagar_zscored_pca_mean =df_sagar_zscored_pca_mean.groupby(['CID']).mean().reset_index()
    
    # df_mean_reduced_sagar_zscored_cid_combined =df_sagar_zscored_pca.groupby(['CID']).mean().reset_index()
#     df_sagar_zscored_pca_mean['Combined']=df_sagar_zscored_pca_mean.loc[:, 0:n_components-1].values.tolist()
#     df_sagar_zscored_pca_mean=df_sagar_zscored_pca_mean.drop([0,1,2,3,4],axis=1)
    
    
    # return df_sagar, df_sagar_mean, df_sagar_zscored, df_sagar_zscored_mean, df_sagar_zscored_pca,df_sagar_zscored_mean_pca,df_sagar_zscored_pca_mean
    
    

    df_sagar_common_mean =df_sagar_common.groupby(['IsomericSMILES','nonStereoSMILES']).apply(lambda x: x.iloc[:, :-4].mean()).reset_index()
    
    
    
    
    columns_list_common = ['Intensity', 'Pleasantness', 'Fishy', 'Burnt', 'Sour', 'Decayed',
       'Musky', 'Fruity', 'Sweaty', 'Cool', 'Floral', 'Sweet',
       'Warm', 'Bakery', 'Spicy']
    
    df_sagar_common_mean['Combined'] = df_sagar_common_mean.loc[:, columns_list_common].values.tolist()
    df_sagar_common['Combined'] = df_sagar_common.loc[:, columns_list_common].values.tolist()
    # return df_sagar_mean
    
#     #Z-score sagar dataset
    df_sagar_common_zscored = df_sagar_common_mean.copy()
    df_sagar_common_zscored=df_sagar_common_zscored.drop('Combined',axis=1)
    scaled_features = StandardScaler().fit_transform(df_sagar_common_zscored.loc[:,columns_list_common].values.tolist())
    df_sagar_common_zscored.loc[:, columns_list_common] = pd.DataFrame(scaled_features, index=df_sagar_common_zscored.index, columns=columns_list_common)
    
#     #Mean over z-score sagar
    df_sagar_common_zscored_mean =df_sagar_common_zscored.groupby(['IsomericSMILES','nonStereoSMILES']).mean().reset_index()
    
    #combine columns
    df_sagar_common_zscored['Combined'] = df_sagar_common_zscored.loc[:, columns_list_common].values.tolist()
    df_sagar_common_zscored_mean['Combined'] = df_sagar_common_zscored_mean.loc[:, columns_list_common].values.tolist()
    
    
    

    return df_sagar, df_sagar_mean, df_sagar_zscored, df_sagar_zscored_mean, df_sagar_common,df_sagar_common_mean, df_sagar_common_zscored,df_sagar_common_zscored_mean


def prepare_sagar_mols(modeldeepchem_gslf,lm,tokenizer):
    # df_sagar=df_sagar.rename(columns={"cid":"CID"})
    df_sagar_mols = df_sagar.drop_duplicates("CID")
    print(df_sagar_mols.columns)
    df_sagar_mols_embeddings_original,df_sagar_mols_layers_original,df_sagar_mols_embeddings,df_sagar_mols_embeddings_zscored,df_sagar_mols_layers,df_sagar_mols_layers_zscored,df_sagar_mols_embeddings_linear,df_sagar_mols_embeddings_linear_zscored=prepare_mols_helper(modeldeepchem_gslf,lm,tokenizer,df_sagar_mols,mol_type="nonStereoSMILES")
    return df_sagar_mols,df_sagar_mols_embeddings_original,df_sagar_mols_layers_original,df_sagar_mols_embeddings,df_sagar_mols_embeddings_zscored,df_sagar_mols_layers,df_sagar_mols_layers_zscored,df_sagar_mols_embeddings_linear,df_sagar_mols_embeddings_linear_zscored






def prepare_snitz_mols(df_snitz_mean,modeldeepchem_gslf,lm,tokenizer):
    df_snitz_mean_mols1 = df_snitz_mean[['Stimulus 1-IsomericSMILES','Stimulus 1-nonStereoSMILES','CID Stimulus 1']].drop_duplicates().reset_index(drop=True)
    df_snitz_mean_mols2 = df_snitz_mean[['Stimulus 2-IsomericSMILES','Stimulus 2-nonStereoSMILES','CID Stimulus 2']].drop_duplicates().reset_index(drop=True).rename(columns={'Stimulus 2-nonStereoSMILES': 'Stimulus 1-nonStereoSMILES','Stimulus 2-IsomericSMILES':'Stimulus 1-IsomericSMILES', 'CID Stimulus 2': 'CID Stimulus 1' })
    df_snitz_mols= pd.concat([df_snitz_mean_mols1, df_snitz_mean_mols2], ignore_index=True, axis=0).reset_index(drop=True)
    df_snitz_mols = df_snitz_mols.rename(columns={'Stimulus 1-IsomericSMILES': 'IsomericSMILES','Stimulus 1-nonStereoSMILES':'nonStereoSMILES', 'CID Stimulus 1': 'CID' })

    df_snitz_mols=df_snitz_mols.drop_duplicates().reset_index(drop=True)
    # df_snitz_mols.to_csv('df_snitz_mols.csv')  
    # mol_type="nonStereoSMILES"
    
    
    df_snitz_mols_embeddings_original,df_snitz_mols_layers_original,\
    df_snitz_mols_embeddings,df_snitz_mols_embeddings_zscored,df_snitz_mols_layers,\
    df_snitz_mols_layers_zscored,df_snitz_mols_embeddings_linear,\
    df_snitz_mols_embeddings_linear_zscored=prepare_mols_helper(modeldeepchem_gslf,lm,tokenizer,df_snitz_mols)
        
    
    return df_snitz_mols,df_snitz_mols_embeddings_original,df_snitz_mols_layers_original,df_snitz_mols_embeddings,df_snitz_mols_embeddings_zscored,df_snitz_mols_layers,df_snitz_mols_layers_zscored,df_snitz_mols_embeddings_linear,df_snitz_mols_embeddings_linear_zscored

def select_features(input_file):
    ds_alva = pd.read_csv(input_file)


    chemical_features_r=["nCIR",
                     "ZM1", 
                     "GNar", 
                     "S1K", 
                     "piPC08",
                     "MATS1v",
                     "MATS7v",
                     "GATS1v", 
                     "Eig05_AEA(bo)", 
                     "SM02_AEA(bo)",
                     "SM03_AEA(dm)",
                     "SM10_AEA(dm)",
                     "SM13_AEA(dm)",
                      "SpMin3_Bh(v)",
                     "RDF035v",
                     "G1m",
                     "G1v",
                     "G1e",
                     "G3s",
                     "R8u+",
                     "nRCOSR"]

    nonStereoSMILE = list(map(lambda x: "nonStereoSMILES___" + x, chemical_features_r))
    # IsomericSMILES = list(map(lambda x: "IsomericSMILES___" + x, chemical_features_r))
    selected_features = nonStereoSMILE
    features= ['CID','nonStereoSMILES']+selected_features
    # print("cc1", ds_alva.columns.values.tolist())
    ds_alva= ds_alva.rename(columns={"cid":"CID"})
    # print("cc2", ds_alva.columns.values.tolist())
    ds_alva_selected = ds_alva[features]
    ds_alva_selected = ds_alva_selected.fillna(0)
    ds_alva_selected['embeddings'] = ds_alva_selected[selected_features].values.tolist()
    return ds_alva_selected

def prepare_mols_other(input_file_embeddings, df_mean,modeldeepchem_gslf):
    df_mean_mols1 = df_mean[['Stimulus 1-IsomericSMILES','Stimulus 1-nonStereoSMILES','CID Stimulus 1']].drop_duplicates().reset_index(drop=True)
    df_mean_mols2 = df_mean[['Stimulus 2-IsomericSMILES','Stimulus 2-nonStereoSMILES','CID Stimulus 2']].drop_duplicates().reset_index(drop=True).rename(columns={'Stimulus 2-nonStereoSMILES': 'Stimulus 1-nonStereoSMILES','Stimulus 2-IsomericSMILES':'Stimulus 1-IsomericSMILES', 'CID Stimulus 2': 'CID Stimulus 1' })
    df_mols= pd.concat([df_mean_mols1, df_mean_mols2], ignore_index=True, axis=0).reset_index(drop=True)
    df_mols = df_mols.rename(columns={'Stimulus 1-IsomericSMILES': 'IsomericSMILES','Stimulus 1-nonStereoSMILES':'nonStereoSMILES', 'CID Stimulus 1': 'CID' })

    df_mols=df_mols.drop_duplicates().reset_index(drop=True)
    # df_snitz_mols.to_csv('df_snitz_mols.csv')  
    mol_type="nonStereoSMILES"
    
    df_embeddigs = pd.read_csv(input_file_embeddings)[['embeddings','CID']]
    # df_embeddigs['embeddings'] = df_embeddigs['embeddings'].apply(ast.literal_eval)
    # df_mols_embeddings_original =[torch.from_numpy(np.asarray(df_embeddigs['embeddings'].values.tolist()))]




    df_embeddigs['embeddings'] = df_embeddigs['embeddings'].apply(lambda x: np.array(eval(x)))


    df_mols['Stimulus Embedding Sum'] = df_mols['CID'].apply(lambda x: sum_embeddings(list(map(int, x.split(','))), df_embeddigs))
    df_mols_embeddings_original =[torch.from_numpy(np.asarray(df_mols['Stimulus Embedding Sum'].values.tolist()))]


    # df_ravia_mols.to_csv('df_ravia_mols.csv')  

    
    
    # df_embeddigs['embeddings'] = df_embeddigs['embeddings'].apply(lambda x: np.array(eval(x)))


    # df_mols['Stimulus Embedding'] = df_mols['CID'].apply(lambda x: extract_embeddings(map(int, x)), df_embeddigs)
    # df_mols_embeddings_original =[torch.from_numpy(np.asarray(df_mols['Stimulus Embedding'].values.tolist()))]


    # df_ravia_mols.to_csv('df_ravia_mols.csv')  


    # df_mols_embeddings_original, df_mols_layers_original=embed(lm,df_ravia_mols[mol_type], tokenizer, batch_size=64)

    df_mols_embeddings_original,df_mols_embeddings,df_mols_embeddings_zscored=prepare_mols_helper_mixture(modeldeepchem_gslf,df_mols_embeddings_original,df_mols)
    # df_mols_embeddings_original =[torch.from_numpy(np.asarray(df_mols_embeddings_original['embeddings'].values.tolist()))]
    
    return df_mols,df_mols_embeddings_original,df_mols_embeddings,df_mols_embeddings_zscored
    # df_mols,df_mols_embeddings_original,df_mols_embeddings,df_mols_embeddings_zscored


def prepare_mols_DAM(input_file_embeddings, df_mean,modeldeepchem_gslf,sep=';'):
    df_mean_mols1 = df_mean[['Stimulus 1-IsomericSMILES','Stimulus 1-nonStereoSMILES','CID Stimulus 1']].drop_duplicates().reset_index(drop=True)
    df_mean_mols2 = df_mean[['Stimulus 2-IsomericSMILES','Stimulus 2-nonStereoSMILES','CID Stimulus 2']].drop_duplicates().reset_index(drop=True).rename(columns={'Stimulus 2-nonStereoSMILES': 'Stimulus 1-nonStereoSMILES','Stimulus 2-IsomericSMILES':'Stimulus 1-IsomericSMILES', 'CID Stimulus 2': 'CID Stimulus 1' })
    df_mols= pd.concat([df_mean_mols1, df_mean_mols2], ignore_index=True, axis=0).reset_index(drop=True)
    df_mols = df_mols.rename(columns={'Stimulus 1-IsomericSMILES': 'IsomericSMILES','Stimulus 1-nonStereoSMILES':'nonStereoSMILES', 'CID Stimulus 1': 'CID' })

    df_mols=df_mols.drop_duplicates().reset_index(drop=True)
    # df_snitz_mols.to_csv('df_snitz_mols.csv')  
    mol_type="nonStereoSMILES"
    
    df_embeddigs = select_features(input_file_embeddings)[['embeddings','CID']]
    # df_embeddigs['embeddings'] = df_embeddigs['embeddings'].apply(ast.literal_eval)
    # df_embeddigs['embeddings'] = df_embeddigs['embeddings'].apply(lambda x: np.array(eval(x)))


    df_mols['Stimulus Embedding'] = df_mols['CID'].apply(lambda x: sum_embeddings(list(map(int, x.split(sep))), df_embeddigs))
    df_mols_embeddings_original =[torch.from_numpy(np.asarray(df_mols['Stimulus Embedding'].values.tolist()))]


    # df_ravia_mols.to_csv('df_ravia_mols.csv')  


    # df_mols_embeddings_original, df_mols_layers_original=embed(lm,df_ravia_mols[mol_type], tokenizer, batch_size=64)

    df_mols_embeddings_original,df_mols_embeddings,df_mols_embeddings_zscored=prepare_mols_helper_mixture(modeldeepchem_gslf,df_mols_embeddings_original,df_mols,last='20')
    
    return df_mols,df_mols_embeddings_original,df_mols_embeddings,df_mols_embeddings_zscored
        
    
    # return df_snitz_mols,df_snitz_mols_embeddings_original,df_snitz_mols_layers_original,df_snitz_mols_embeddings,df_snitz_mols_embeddings_zscored,df_snitz_mols_layers,df_snitz_mols_layers_zscored,df_snitz_mols_embeddings_linear,df_snitz_mols_embeddings_linear_zscored

    
    # return df_ravia_mols



def prepare_goodscentleffignwell_mols(modeldeepchem_gslf,lm,tokenizer):
    goodscentleffignwell_input_file = '/local_storage/datasets/farzaneh/alignment_olfaction_datasets/curated_datasets/mols_datasets/curated_GS_LF_merged_4983.csv' # or new downloaded file path
    df_goodscentleffignwell=pd.read_csv(goodscentleffignwell_input_file)
    df_goodscentleffignwell.index.names = ['CID']
    # return df
    df_goodscentleffignwell_mols_layers=[]
    df_goodscentleffignwell_mols_layers_zscored=[]
    
    # input_file = '/local_storage/datasets/farzaneh/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv' # or new downloaded file path
    df_goodscentleffignwell= df_goodscentleffignwell.reset_index()
  
    df_goodscentleffignwell['y'] = df_goodscentleffignwell.loc[:,'alcoholic':'woody'].values.tolist()
    
#      #inference on molecules
    df_gslf_mols_embeddings_original,df_gslf_mols_layers_original,df_gslf_mols_embeddings,df_gslf_mols_embeddings_zscored,df_gslf_mols_layers,df_gslf_mols_layers_zscored,df_gslf_mols_embeddings_linear,df_gslf_mols_embeddings_linear_zscored=prepare_mols_helper(modeldeepchem_gslf,lm,tokenizer,df_goodscentleffignwell)
#     df_snitz_mols_embeddings_original,df_snitz_mols_layers_original,df_snitz_mols_embeddings,df_snitz_mols_embeddings_zscored,df_snitz_mols_layers,df_snitz_mols_layers_zscored=prepare_mols_helper(df_snitz_mols)
    return df_goodscentleffignwell, df_gslf_mols_embeddings_original,df_gslf_mols_layers_original,df_gslf_mols_embeddings,df_gslf_mols_embeddings_zscored,df_gslf_mols_layers,df_gslf_mols_layers_zscored,df_gslf_mols_embeddings_linear,df_gslf_mols_embeddings_linear_zscored


    
#     return df_snitz_mols,df_snitz_mols_embeddings_original,df_snitz_mols_layers_original,df_snitz_mols_embeddings,df_snitz_mols_embeddings_zscored,df_snitz_mols_layers,df_snitz_mols_layers_zscored

