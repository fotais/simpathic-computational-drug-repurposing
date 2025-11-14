import torch
import torch.nn.functional as F
import pandas as pd
from sentence_transformers import SentenceTransformer
import json

rel_no=70
path="/home/faisopos/workspace/simpathic/new-min-db-noDB/"

class SequenceEncoder(object):
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values.astype(str), show_progress_bar=True, convert_to_tensor=True, device=self.device)
        return x.cpu()



class TypesEncoder(object):
    def __init__(self, sep='|'):
        self.sep = sep

    def __call__(self, df):
        genres = set(g for col in df.values for g in col.split(self.sep))
        mapping = {genre: i for i, genre in enumerate(genres)}

        x = torch.zeros(len(df), len(mapping))
        for i, col in enumerate(df.values):
            for genre in col.split(self.sep):
                x[i, mapping[genre]] = 1
        return x

class IdentityEncoder(object):
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)

def normaliseRelation(rel):
    
    if rel=="ADMINISTERED_TO":
        norm_rel="ADMINISTERED_TO"
    elif rel=="ADMINISTERED_TO__SPEC__":
        norm_rel= "ADMINISTERED_TO"
    elif rel=="AFFECTS":
        norm_rel= "AFFECTS"        
    elif rel=="AFFECTS__SPEC__":
        norm_rel= "AFFECTS"
    elif rel=="ASSOCIATED_WITH":
        norm_rel= "ASSOCIATED_WITH"
    elif rel=="ASSOCIATED_WITH__INFER__":
        norm_rel= "ASSOCIATED_WITH"
    elif rel=="ASSOCIATED_WITH__SPEC__":
        norm_rel= "ASSOCIATED_WITH"
    elif rel=="AUGMENTS":
        norm_rel= "AUGMENTS"
    elif rel=="AUGMENTS__SPEC__":
        norm_rel= "AUGMENTS"
    elif rel=="CAUSES":
        norm_rel= "CAUSES"
    elif rel=="CAUSES__SPEC__":
        norm_rel= "CAUSES"
    elif rel=="COEXISTS_WITH":
        norm_rel= "COEXISTS_WITH"
    elif rel=="COEXISTS_WITH__SPEC__":
        norm_rel= "COEXISTS_WITH"
    elif rel=="compared_with":
        norm_rel= "compared_with"
    elif rel=="compared_with__SPEC__":
        norm_rel= "compared_with"
    elif rel=="COMPLICATES":
        norm_rel= "COMPLICATES"
    elif rel=="COMPLICATES__SPEC__":
        norm_rel= "COMPLICATES"
    elif rel=="CONVERTS_TO":
        norm_rel= "CONVERTS_TO"
    elif rel=="CONVERTS_TO__SPEC__":
        norm_rel= "CONVERTS_TO"
    elif rel=="DIAGNOSES":
        norm_rel= "DIAGNOSES"
    elif rel=="DIAGNOSES__SPEC__":
        norm_rel= "DIAGNOSES"
    elif rel=="different_from":
        norm_rel= "different_from"
    elif rel=="different_from__SPEC__":
        norm_rel= "different_from"
    elif rel=="different_than":
        norm_rel= "different_than"
    elif rel=="different_than__SPEC__":
        norm_rel= "different_than"
    elif rel=="DISRUPTS":
        norm_rel= "DISRUPTS"
    elif rel=="DISRUPTS__SPEC__":
        norm_rel= "DISRUPTS"
    elif rel=="higher_than":
        norm_rel= "higher_than"
    elif rel=="higher_than__SPEC__":
        norm_rel= "higher_than"
    elif rel=="INHIBITS":
        norm_rel= "INHIBITS"
    elif rel=="INHIBITS__SPEC__":
        norm_rel= "INHIBITS"
    elif rel=="INTERACTS_WITH":
        norm_rel= "INTERACTS_WITH"
    elif rel=="INTERACTS_WITH__INFER__":
        norm_rel= "INTERACTS_WITH"
    elif rel=="INTERACTS_WITH__SPEC__":
        norm_rel= "INTERACTS_WITH"
    elif rel=="IS_A":
        norm_rel= "IS_A"
    elif rel=="ISA":
        norm_rel= "ISA"
    elif rel=="LOCATION_OF":
        norm_rel= "LOCATION_OF"
    elif rel=="LOCATION_OF__SPEC__":
        norm_rel= "LOCATION_OF"
    elif rel=="lower_than":
        norm_rel= "lower_than"
    elif rel=="lower_than__SPEC__":
        norm_rel= "lower_than"
    elif rel=="MANIFESTATION_OF":
        norm_rel= "MANIFESTATION_OF"
    elif rel=="MANIFESTATION_OF__SPEC__":
        norm_rel= "MANIFESTATION_OF"
    elif rel=="METHOD_OF":
        norm_rel= "METHOD_OF"
    elif rel=="METHOD_OF__SPEC__":
        norm_rel= "METHOD_OF"
    elif rel=="OCCURS_IN":
        norm_rel= "OCCURS_IN"
    elif rel=="OCCURS_IN__SPEC__":
        norm_rel= "OCCURS_IN"
    elif rel=="PART_OF":
        norm_rel= "PART_OF"
    elif rel=="PART_OF__SPEC__":
        norm_rel= "PART_OF"
    elif rel=="PRECEDES":
        norm_rel= "PRECEDES"
    elif rel=="PRECEDES__SPEC__":
        norm_rel= "PRECEDES"
    elif rel=="PREDISPOSES":
        norm_rel= "PREDISPOSES"
    elif rel=="PREDISPOSES__SPEC__":
        norm_rel= "PREDISPOSES"
    elif rel=="PREVENTS":
        norm_rel= "PREVENTS"
    elif rel=="PREVENTS__SPEC__":
        norm_rel= "PREVENTS"
    elif rel=="PROCESS_OF":
        norm_rel= "PROCESS_OF"
    elif rel=="PROCESS_OF__SPEC__":
        norm_rel= "PROCESS_OF"
    elif rel=="PRODUCES":
        norm_rel= "PRODUCES"
    elif rel=="PRODUCES__SPEC__":
        norm_rel= "PRODUCES"
    elif rel=="same_as":
        norm_rel= "same_as"
    elif rel=="same_as__SPEC__":
        norm_rel= "same_as"
    elif rel=="STIMULATES":
        norm_rel= "STIMULATES"
    elif rel=="STIMULATES__SPEC__":
        norm_rel= "STIMULATES"
    elif rel=="IS_TREATED":
        norm_rel= "IS_TREATED"
    elif rel=="USES":
        norm_rel= "USES"
    elif rel=="USES__SPEC__":
        norm_rel= "USES"
    elif rel=="MENTIONED_IN":
        norm_rel= "MENTIONED_IN"
    elif rel=="HAS_MESH":
        norm_rel= "HAS_MESH"
    else:
        norm_rel="ASSOCIATED_WITH"

    return norm_rel

def encodeEdgeTypes(df):
    reltypes = ["ADMINISTERED_TO","AFFECTS","ASSOCIATED_WITH","AUGMENTS","CAUSES","COEXISTS_WITH","compared_with","COMPLICATES","CONVERTS_TO","DIAGNOSES","different_from","different_than","DISRUPTS","higher_than","INHIBITS","INTERACTS_WITH","IS_A","ISA","LOCATION_OF","lower_than","MANIFESTATION_OF","METHOD_OF","OCCURS_IN","PART_OF","PRECEDES","PREDISPOSES","PREVENTS","PROCESS_OF","PRODUCES","same_as","STIMULATES","IS_TREATED","USES","MENTIONED_IN","HAS_MESH"]
    mapping = {rtype: i for i, rtype in enumerate(reltypes)}
    x = torch.zeros(len(df), dtype=torch.int64)
    for i, col in enumerate(df.values):
        rel=normaliseRelation(col)
#        print('edgetype i, x[i]: ',i, type(mapping[rel]))
        x[i]= mapping[rel]
    return x



def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, sep='\t', **kwargs)

    mapping = {index: i for i, index in enumerate(df.index.unique())}
    
    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping


def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    df = pd.read_csv(path, sep='\t', **kwargs)
    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
#        print('edge_attrs: ', edge_attrs)
        edge_attr = torch.cat(edge_attrs, dim=-1)
#        print('CONCATENATED edge_attr: ', edge_attr)
    return edge_index, edge_attr


def extractSimilaritiesFromJSON():
    similarities = {}
    # Opening JSON file
    with open('./workspace/simpathic/disease2disease_cuis.json') as json_file:
        data = json.load(json_file)

        for key, value in data.items():
            for item in value:
                #print(" syndrome1=", key," syndrome2=", item[1], "similarity=",item[0])
                ###Save the cuis (int values) of two syndromes, along with their similarity value in a dictionary
                intcui1 = convertCUItoINT(key)
                intcui2 = convertCUItoINT(item[1])
                similarities[(intcui1, intcui2)] = item[0]           
    return similarities

def extractDisCUIsFromJSON():
    cuis = []
    with open('./workspace/simpathic/disease2disease_cuis.json') as json_file:
        data = json.load(json_file)
        for key, value in data.items():
            intcui = convertCUItoINT(key)
            cuis.append(intcui)           
    return cuis

def convertCUItoINT(cui):
    cui=cui.replace('C', '')
    intcui = int(cui)
    return intcui
    
def load_entity_feats (path, index_col, **kwargs):
    df = pd.read_csv(path, index_col=index_col, sep='\t', **kwargs)
    #####ADDITION
    disease_cuis = extractDisCUIsFromJSON()
    dis_nodes = df.copy()
    dis_nodes = dis_nodes.drop(columns=['SEM_TYPES'])
    dis_nodes = dis_nodes.loc[dis_nodes['CUI'].isin(disease_cuis)]
    print('dis_nodes=', dis_nodes)
    dis_similarities = extractSimilaritiesFromJSON()
    #print('dis_similarities=', dis_similarities)
    #####
    
    df_semTypes=df['SEM_TYPES']
    #print('type(df_semTypes)=',type(df_semTypes))
    typesep="|"
    genres = set(g for col in df_semTypes.values for g in col.split(typesep))
    mapping = {genre: i for i, genre in enumerate(genres)}
    #print(' len(mapping)=', len(mapping)) # 126
    x = torch.zeros(len(df_semTypes), len(mapping)+1)
    for i, col in enumerate(df_semTypes.values):
        for genre in col.split(typesep):
            x[i, mapping[genre]] = 1
    return x, dis_nodes, dis_similarities
  

def load_article_feats (path,  **kwargs):
    df = pd.read_csv(path, sep='\t', **kwargs)
    
    x = torch.zeros(len(df), 127)
    for i, j in enumerate(x):
        x[i, 126] = 1.0
    return x       

################################################
###################Load CSV#####################
################################################

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#print(torch.version.cuda)

ent_file_path=path+"entities.tsv"
rel_file_path=path+"relations.tsv"
rel1_file_path=path+"relations_entities.tsv"
rel2_file_path=path+"relations_articles.tsv"
art_file_path=path+"articles.tsv"


entity_cui, entity_type = load_node_csv(
    ent_file_path, index_col='ID', encoders={
        'CUI': IdentityEncoder(dtype=torch.long),
        'SEM_TYPES': TypesEncoder()
    })

article_title, article_no = load_node_csv(
    art_file_path, index_col='AID', encoders={
        'TITLE': SequenceEncoder(),
        'ARTICLE_NO': IdentityEncoder(dtype=torch.int)
    })

from torch_geometric.data import HeteroData


data = HeteroData()

data['entity'].x, dis_nodes, dis_similarities = load_entity_feats(ent_file_path, index_col='ID') #entity_cui
data['article'].x = load_article_feats(art_file_path) #article_title

data['entity'].num_nodes= 162020 #!!!!!with DRUGBANK: 169857
#print('!!!!!!!!!!entity_cui=', entity_cui)
#print('!!!!!!!!!!entity_type=', entity_type)
#print('!!!!!!!!!!data[entity].num_nodes=', data['entity'].num_nodes)
#print('!!!!!!!!!!data[article].num_nodes=', data['article'].num_nodes)
#print('!!!!!!!!!!data.num_nodes=', data.num_nodes)

#Insert and Encode relations

print('load entity-entity rels')

edge_index, edge_attr = load_edge_csv(
    rel1_file_path,
    src_index_col='NOD1',
    src_mapping=entity_type,
    dst_index_col='NOD2',
    dst_mapping=entity_type,
    encoders={'REFERENCES': IdentityEncoder(dtype=torch.long)#,  'RELATION': InteractsEncoder() 
},
)

data['entity', 'rel', 'entity'].edge_index = edge_index
data['entity', 'rel', 'entity'].edge_attr = edge_attr


print('load article-entity rels')

article_no_int = {}
for key, value in article_no.items():
    try:
        article_no_int[int(float(key))] = value
    except ValueError:
        print("article no error")#print("Not a float"+ " "+ str(key)+" "+str(value))

edge_index, edge_attr = load_edge_csv(
    rel2_file_path,
    src_index_col='NOD1',
    src_mapping=article_no_int,
    dst_index_col='NOD2',
    dst_mapping=entity_type,
    encoders={'REFERENCES': IdentityEncoder(dtype=torch.long)
#          'RELATION': InteractsEncoder()
    },
)

data['article', 'rel', 'entity'].edge_index = edge_index
data['article', 'rel', 'entity'].edge_attr = edge_attr

df = pd.read_csv(rel1_file_path, sep='\t')
df_rel=df["RELATION"]
edge_type = encodeEdgeTypes(df_rel)


################################################
###################GCN##########################
################################################


""""
Implements the link prediction task on the FB15k237 datasets according to the
`"Modeling Relational Data with Graph Convolutional Networks"
<https://arxiv.org/abs/1703.06103>`_ paper.
Caution: This script is executed in a full-batch fashion, and therefore needs
to run on CPU (following the experimental setup in the official paper).
"""

from torch.nn import Parameter
from tqdm import tqdm

from torch_geometric.datasets import RelLinkPredDataset
from torch_geometric.nn import GAE, RGCNConv


class RGCNEncoder(torch.nn.Module):
    def __init__(self, num_nodes, hidden_channels, num_relations):
        super().__init__()
        self.node_emb = Parameter(torch.Tensor(num_nodes, hidden_channels))
        self.conv1 = RGCNConv(hidden_channels, hidden_channels, num_relations,
                              num_blocks=5)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations,
                              num_blocks=5)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.node_emb)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
  
    def forward(self, edge_index, edge_type):
        x = self.node_emb
        x = self.conv1(x, edge_index, edge_type).relu_()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        #print('edge_index[0]=', edge_index[0])
        #print('edge_type[0]=', edge_type[0])
        
        ############ADDITION for dis similarities############
        #print("###FW function run - SIMILARITIES x_sim calculation###")
        x_sim = torch.zeros(198387, 100)
        num_of_diseases = len(dis_nodes)
        i=0
        for index, row in  dis_nodes.iterrows():
            i=i+1
            #print("index=", index, "intcui=",row['CUI'])
            for index2, row2 in  dis_nodes.iterrows():
                if (index==index2):
                    continue
                simil=dis_similarities[(row['CUI'],row2['CUI'])]
                simil_num = float(simil)
                x_sim[index]=x_sim[index]+simil_num*x[index2]
                #if i<2:
                 #   print("x_sim[index]: ",x_sim[index])
        x_sim=x_sim/(float(num_of_diseases)-1.0)
        for index, row in  dis_nodes.iterrows():
            x[index]=(x[index]+x_sim[index])/2.0
        #print("###FINISHED FW function run###")
        #x=x+x_sim
        ############       END OF ADDITION       ############
            
        return x



class DistMultDecoder(torch.nn.Module):
    def __init__(self, num_relations, hidden_channels):
        super().__init__()
        self.rel_emb = Parameter(torch.Tensor(num_relations, hidden_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.rel_emb)

    def forward(self, z, edge_index, edge_type):
        z_src, z_dst = z[edge_index[0]], z[edge_index[1]]
        rel = self.rel_emb[edge_type]
        return torch.sum(z_src * rel * z_dst, dim=1)


model = GAE(
    RGCNEncoder(data.num_nodes, hidden_channels=100,
                num_relations=rel_no),
    DistMultDecoder(rel_no // 2, hidden_channels=100),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    #print('Now training the RGCN model:')
    model.train()
    optimizer.zero_grad()

    #print('RGCNEncoder running...')  
    z = model.encode(data['entity', 'rel', 'entity'].edge_index, edge_type)
    #print('Decoder running...')
    pos_out = model.decode(z, train_edge_index, train_edge_type)

    #neg_edge_index = negative_sampling(train_edge_index, data.num_nodes)
    #neg_out = model.decode(z, neg_edge_index, train_edge_type)
    neg_out = model.decode(z, orig_neg_edge_index, orig_neg_edge_type)

    out = torch.cat([pos_out, neg_out])
    gt = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
    cross_entropy_loss = F.binary_cross_entropy_with_logits(out, gt)
    reg_loss = z.pow(2).mean() + model.decoder.rel_emb.pow(2).mean()
    loss = cross_entropy_loss + 1e-2 * reg_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()

    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    z = model.encode(data['entity', 'rel', 'entity'].edge_index, edge_type)

##    test_mrr = compute_mrr(z, test_edge_index, test_edge_type)
##    mrr=test_mrr.item()
##    print('test_mrr=', mrr)
    mrr = query_based_MRR(500, z, all_test_edge_index, all_test_edge_type, test_edge_index)
    print('mrr=', mrr)
    hitsAt1 = query_based_HitsAt(1, z, all_test_edge_index, all_test_edge_type, test_edge_index)
    print('hitsAt1=', hitsAt1)
    hitsAt5 = query_based_HitsAt(5, z, all_test_edge_index, all_test_edge_type, test_edge_index)
    print('hitsAt5=', hitsAt5)
    hitsAt10 = query_based_HitsAt(10, z, all_test_edge_index, all_test_edge_type, test_edge_index)
    print('hitsAt10=', hitsAt10)
    hitsAt100 = query_based_HitsAt(100, z, all_test_edge_index, all_test_edge_type, test_edge_index)
    print('hitsAt100=', hitsAt100)
    
    tps,fns = compute_poss(z, test_edge_index, test_edge_type, 1)
    tns,fps = compute_negs(z, test_neg_edge_index, test_neg_edge_type, 0)
    ###for debug purposes
    ###print("tested ", (tps+fns), "positive pairs and ", (tns+fps), " negative pairs")
    if tps==0 and fps==0:
        return 0,0,0, tps, fps, fns, mrr, hitsAt1, hitsAt5, hitsAt10, hitsAt100
    prec=tps/(tps+fps)
    rec=tps/(tps+fns)
    if prec==0 and rec==0:
        f1=0
    else:
        f1=2*prec*rec/(prec+rec)
    return prec, rec, f1, tps, fps, fns, mrr, hitsAt1, hitsAt5, hitsAt10, hitsAt100 #valid_mrr, test_mrr

from tqdm import tqdm




#####My CUSTOM function for calculating query-based MRR...
@torch.no_grad()
def query_based_MRR(N, z, edge_index, edge_type, p_edge_index):

    ####Get out predictions for all test pairs
    syndromes = edge_index[0,:].tolist()
    uniquesyndromes = list(dict.fromkeys(syndromes))
    drugs = edge_index[1,:].tolist()
    uniquedrugs = list(dict.fromkeys(drugs))
    #print('syndromes=', uniquesyndromes, ' drugs=', uniquedrugs)
    #####Examine rr for each syndrome and average in the end
    rrTotal=0.0
    for query in positivePairs:
        
        rr=0.0
        syndrome_drug = query.split(",")
        syndrome = syndrome_drug[0]
        drug = syndrome_drug[1]

        
        ####for each query {syndrome,drug} filter out the remaining correct drugs
        filteredrugs=uniquedrugs.copy()
        #print('filteredrugs=', filteredrugs)
        for pair in positivePairs:
            syndrome_drug = pair.split(",")
            iter_syndrome = syndrome_drug[0]
            iter_drug = syndrome_drug[1]
            
            if iter_syndrome==syndrome and iter_drug!=drug:
                d = drug_id_to_index[int(iter_drug)]
      #          print('syndrome:', syndrome, ' main drug:', drug, ' remove -->', iter_drug, 'index -->', int(d))
                if int(d) in filteredrugs:
                    filteredrugs.remove(int(d))
      
        ####now create all combinations of syndrome-drugs
        repeatedSyndrom = []
        for r in range(len(filteredrugs)):
            repeatedSyndrom.append(int(syndrome))   
        allcombinations=[repeatedSyndrom,filteredrugs]
       
        test_edge_index = torch.tensor(allcombinations)

        #####create a tensor with one repeated edge_type(TREATED rel type) for prediction
        alltypes = []
        for r in range(len(filteredrugs)):
            alltypes.append(edge_type[0].item())
        test_edge_type = torch.tensor(alltypes)
        #print('test_edge_type=', test_edge_type)
            
        out = model.decode(z, test_edge_index, test_edge_type)
        #    print('out=', out)
            
        #####Now find first N drugs in ranking and save their indexes in a list
        top_n_indexes = []
        for r in range(N):
            m = torch.max(out)
            i = torch.argmax(out)
                #if r<5:
                #    print('r=', r, ' found max=', m, '  in index=', i.item())#, 'drug id=', df_neg_test["NOD2"][i.item()])        
            top_n_indexes.append(i)
            out[i]=-1000.0
        top_n_indexes=torch.as_tensor(top_n_indexes)
            #print('top_n_indexes=', top_n_indexes)

            #print('top drugs based on top_n_indexes: ', test_edge_index[1,top_n_indexes.item()])
        top_n_drugs = torch.index_select(test_edge_index[1,:], 0, top_n_indexes)
            #print('top_n_drug_indexes=', top_n_drugs)
            #print('Look in p_edges index=', p_edge_index)
            
        rank=0.0
        drug_id = drug_id_to_index[int(drug)]
        #print('Look for drug: ', drug_id, ' in ranked list:')
        for dr_ind in top_n_drugs:
            rank = rank+1.0   
            d = dr_ind.item() #drug_index_to_id[test_edge_index[i.item(),1]]
            #print('Drug ', d, ' at rank: ', rank)
            if d==drug_id:
                print('@@@ HIT! drug ', d, 'found for syndrome ', syndrome, 'in positive pairs. Ranking at place: ', rank)
                rr=1.0/rank
                break
            
            ####
            
                
        rrTotal=rrTotal+ rr
        
    mrr=float(rrTotal)/float(len(positivePairs))
    return mrr
######


@torch.no_grad()
def query_based_HitsAt(N, z, edge_index, edge_type, p_edge_index):

    ####Get out predictions for all test pairs
    syndromes = edge_index[0,:].tolist()
    uniquesyndromes = list(dict.fromkeys(syndromes))
    drugs = edge_index[1,:].tolist()
    uniquedrugs = list(dict.fromkeys(drugs))
    #print('syndromes=', uniquesyndromes, ' drugs=', uniquedrugs)
    #####Examine rr for each syndrome and average in the end
    hitsAtNTotal=0.0
    for query in positivePairs:
        
        syndrome_drug = query.split(",")
        syndrome = syndrome_drug[0]
        drug = syndrome_drug[1]

 
        ####for each query {syndrome,drug} filter out the remaining correct drugs
        filteredrugs=uniquedrugs.copy()
        for pair in positivePairs:
            syndrome_drug = pair.split(",")
            iter_syndrome = syndrome_drug[0]
            iter_drug = syndrome_drug[1]
            
            if iter_syndrome==syndrome and iter_drug!=drug:
                d = drug_id_to_index[int(iter_drug)]
                if int(d) in filteredrugs:
                    filteredrugs.remove(int(d))
    
        ####now create all combinations of syndrome-drugs
        repeatedSyndrom = []
        for r in range(len(filteredrugs)):
            repeatedSyndrom.append(int(syndrome))   
        allcombinations=[repeatedSyndrom,filteredrugs]       

        test_edge_index = torch.tensor(allcombinations)
 
        #####create a tensor with one repeated edge_type(TREATED rel type) for prediction
        alltypes = []
        for r in range(len(filteredrugs)):
            alltypes.append(edge_type[0].item())
        test_edge_type = torch.tensor(alltypes)
        #print('test_edge_type=', test_edge_type)
            
        out = model.decode(z, test_edge_index, test_edge_type)
        #    print('out=', out)
            
        #####Now find first N drugs in ranking and save their indexes in a list
        top_n_indexes = []
        for r in range(N):
            m = torch.max(out)
            i = torch.argmax(out)
                #if r<5:
                #    print('r=', r, ' found max=', m, '  in index=', i.item())#, 'drug id=', df_neg_test["NOD2"][i.item()])        
            top_n_indexes.append(i)
            out[i]=-1000.0
        top_n_indexes=torch.as_tensor(top_n_indexes)
            #print('top_n_indexes=', top_n_indexes)

            #print('top drugs based on top_n_indexes: ', test_edge_index[1,top_n_indexes.item()])
        top_n_drugs = torch.index_select(test_edge_index[1,:], 0, top_n_indexes)
            #print('top_n_drug_indexes=', top_n_drugs)
            #print('Look in p_edges index=', p_edge_index)

       
        #####Search top_n pairs in positive pairs list, to decide a hit
        drug_id = drug_id_to_index[int(drug)]
        for dr_ind in top_n_drugs:
            d = dr_ind.item() #drug_index_to_id[test_edge_index[i.item(),1]]
            if drug_id == d:
                hitsAtNTotal=hitsAtNTotal+1
                break
        
        
    hitsAtN=float(hitsAtNTotal)/float(len(positivePairs))
    return hitsAtN
######


@torch.no_grad()
def compute_mrr(z, edge_index, edge_type):
    ranks = []
    for i in tqdm(range(edge_type.numel())):
        (src, dst), rel = edge_index[:, i], edge_type[i]
        # Try all nodes as tails, but delete true triplets:
        tail_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        for (heads, tails), types in [
            (train_edge_index, train_edge_type),
#            (data.valid_edge_index, data.valid_edge_type),
            (test_edge_index, test_edge_type),
        ]:
            tail_mask[tails[(heads == src) & (types == rel)]] = False

        tail = torch.arange(data.num_nodes)[tail_mask]
        tail = torch.cat([torch.tensor([dst]), tail])
        head = torch.full_like(tail, fill_value=src)
        eval_edge_index = torch.stack([head, tail], dim=0)
        eval_edge_type = torch.full_like(tail, fill_value=rel)

        out = model.decode(z, eval_edge_index, eval_edge_type)
        rank = compute_rank(out)
        ranks.append(rank)
    

        # Try all nodes as heads, but delete true triplets:
        head_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        for (heads, tails), types in [
            (train_edge_index, train_edge_type),
 #           (data.valid_edge_index, data.valid_edge_type),
            (test_edge_index, test_edge_type),
        ]:
            head_mask[heads[(tails == dst) & (types == rel)]] = False

        head = torch.arange(data.num_nodes)[head_mask]
        head = torch.cat([torch.tensor([src]), head])
        tail = torch.full_like(head, fill_value=dst)
        eval_edge_index = torch.stack([head, tail], dim=0)
        eval_edge_type = torch.full_like(head, fill_value=rel)

        out = model.decode(z, eval_edge_index, eval_edge_type)
        rank = compute_rank(out)
        ranks.append(rank)

    return (1. / torch.tensor(ranks, dtype=torch.float)).mean()

@torch.no_grad()
def compute_rank(ranks):
    # fair ranking prediction as the average
    # of optimistic and pessimistic ranking
    true = ranks[0]
    optimistic = (ranks > true).sum() + 1
    pessimistic = (ranks >= true).sum()
    return (optimistic + pessimistic).float() * 0.5



#####My CUSTOM function for calculating disease-based MRR...
@torch.no_grad()
def disease_based_MRR(N, z, edge_index, edge_type, p_edge_index):

    ####Get out predictions for all test pairs
    syndromes = edge_index[0,:].tolist()
    uniquesyndromes = list(dict.fromkeys(syndromes))
    drugs = edge_index[1,:].tolist()
    uniquedrugs = list(dict.fromkeys(drugs))
    #print('syndromes=', uniquesyndromes, ' drugs=', uniquedrugs)
    #####Examine rr for each syndrome and average in the end
    rrTotal=0.0
    for syndrome in uniquesyndromes:
        #print('examine syndrome index: ', syndrome, 'id=', df_test["NOD1"][syndrome])
        rr=0.0
        #####create a tensor with one repeated syndrome and all drugs
        repeatedSyndrom = []
        for r in range(len(uniquedrugs)):
             repeatedSyndrom.append(syndrome)   
        allcombinations=[repeatedSyndrom,uniquedrugs]
        #print('allcombinations=', allcombinations)
        test_edge_index = torch.tensor(allcombinations)
        #####create a tensor with one repeated edge_type(TREATED rel type) for prediction
        alltypes = []
        for r in range(len(uniquedrugs)):
            alltypes.append(edge_type[0].item())
        test_edge_type = torch.tensor(alltypes)
        #print('test_edge_type=', test_edge_type)
        
        out = model.decode(z, test_edge_index, test_edge_type)
    #    print('out=', out)
        
        #####Now find first N drugs in ranking and save their indexes in a list
        top_n_indexes = []
        for r in range(N):
            m = torch.max(out)
            i = torch.argmax(out)
            #if r<5:
            #    print('r=', r, ' found max=', m, '  in index=', i.item())#, 'drug id=', df_neg_test["NOD2"][i.item()])        
            top_n_indexes.append(i)
            out[i]=-1000.0
        top_n_indexes=torch.as_tensor(top_n_indexes)
        #print('top_n_indexes=', top_n_indexes)

        #print('top drugs based on top_n_indexes: ', test_edge_index[1,top_n_indexes.item()])
        top_n_drugs = torch.index_select(test_edge_index[1,:], 0, top_n_indexes)
        #print('top_n_drug_indexes=', top_n_drugs)
        #print('Look in p_edges index=', p_edge_index)
        
        ####WORKS!!!
        rank=0.0
        for dr_ind in top_n_drugs:
            rank = rank+1.0   
            s = dis_index_to_id[syndrome]
            d = drug_index_to_id[dr_ind.item()] #drug_index_to_id[test_edge_index[i.item(),1]]
            pair = str(s)+','+str(d)
            #print('@@@@@@@@@@@@@@@ EXAMINE PAIR: ', pair)
            if pair in positivePairs:
                print('@@@ HIT! drug ', d, 'found for syndrome ', s, 'in positive pairs. Ranking at place: ', rank)
                rr=1.0/rank
                break
        
        ####
            
        rrTotal=rrTotal+ rr
        
    mrr=float(rrTotal)/float(len(uniquesyndromes))
    return mrr
######

@torch.no_grad()
def disease_based_HitsAt(N, z, edge_index, edge_type, p_edge_index):

    ####Get out predictions for all test pairs
    #print('edge_index=', edge_index)
    syndromes = edge_index[0,:].tolist()
    uniquesyndromes = list(dict.fromkeys(syndromes))
    drugs = edge_index[1,:].tolist()
    uniquedrugs = list(dict.fromkeys(drugs))
    #print('syndromes=', uniquesyndromes, ' drugs=', uniquedrugs)
    #####Examine hits@N for each syndrome and average in the end
    hitsAtNTotal=0
    for syndrome in uniquesyndromes:
        #####create a tensor with one repeated syndrome and all drugs
        repeatedSyndrom = []
        for r in range(len(uniquedrugs)):
             repeatedSyndrom.append(syndrome)   
        allcombinations=[repeatedSyndrom,uniquedrugs]
    #    print('allcombinations=', allcombinations)
        test_edge_index = torch.tensor(allcombinations)
    #    print('edge_type=', edge_type)
        #####create a tensor with one repeated edge_type(TREATED rel type) for prediction
        alltypes = []
        for r in range(len(uniquedrugs)):
            alltypes.append(edge_type[0].item())
        test_edge_type = torch.tensor(alltypes)
        
        out = model.decode(z, test_edge_index, test_edge_type)
    #    print('out=', out)
        
        #####Now find first N drugs in ranking and save their indexes in a list
        top_n_indexes = []
        for r in range(N):
            m = torch.max(out)
            i = torch.argmax(out)
    #        print('r=', r, ' found max=', m, '  in index=', i)        
            top_n_indexes.append(i)
            out[i]=-1000.0
        top_n_indexes=torch.as_tensor(top_n_indexes)
    #    print('top_n_indexes=', top_n_indexes)

        #print('top drugs based on top_n_indexes: ', test_edge_index[1,top_n_indexes.item()])
        top_n_drugs = torch.index_select(test_edge_index[1,:], 0, top_n_indexes)
        #print('top_n_drugs=', top_n_drugs)
        hits=0
        
        #####Search top_n pairs in positive pairs list, to decide a hit
        for dr_ind in top_n_drugs:
            s = dis_index_to_id[syndrome]
            d = drug_index_to_id[dr_ind.item()] #drug_index_to_id[test_edge_index[i.item(),1]]
            pair = str(s)+','+str(d)

            if pair in positivePairs:
                hits=1
                break

        #print('For Syndrome: ', syndrome, ' hitsAt', N,':', hits)
        hitsAtNTotal=hitsAtNTotal+  hits
        
    hitsAtN=float(hitsAtNTotal)/float(len(uniquesyndromes))
    return hitsAtN
######



@torch.no_grad()
def compute_negs (z, eval_edge_index, eval_edge_type, pol):
    out = model.decode(z, eval_edge_index, eval_edge_type)
    #print('out object=', out)
    tns=0
    fps=0
    for predi in out:
        if predi>0.0:
            fps=fps+1
        else:
            tns=tns+1
    return tns,fps

@torch.no_grad()
def compute_poss (z, eval_edge_index, eval_edge_type, pol):
    out = model.decode(z, eval_edge_index, eval_edge_type)
    #print('out object=', out)
    tps=0
    fns=0
    for predi in out:
        if predi>0.0:
            tps=tps+1
        else:
            fns=fns+1
    return tps,fns



################################################
################### Groundtruth ################
################################################

groundtruth=path+"posGroundtruth_filtered.tsv"
neg_groundtruth=path+"negGroundtruth_filtered.tsv"

print('load entity-entity TRAIN rels')

ie = IdentityEncoder(dtype=torch.long)

###Assuming that the first 35 and 10929 rows respectively are the test set...
df_train_ground= pd.read_csv(groundtruth, sep='\t', names = ['NOD1','NOD2','RELATION','REFERENCES'], skiprows=34) #!!!!!with DRUGBANK: 35
df_train_neg_ground= pd.read_csv(neg_groundtruth, sep='\t', names = ['NOD1','NOD2','RELATION','REFERENCES'], skiprows=6490) #!!!!!with DRUGBANK: 10929
train_neg_rtypes = df_train_neg_ground["RELATION"]
train_gener_refs = [0]*len(train_neg_rtypes)
df_train_neg_ground['REFERENCES'] = train_gener_refs
df_train = pd.concat([df_train_ground, df_train_neg_ground])
#dfLabels=["NOD1","NOD2","RELATION"]
df_train_pairs_rtypes= df_train.drop('REFERENCES', axis=1)
df_train_refs = df_train["REFERENCES"]

###Skipping last TotalPoss-35 and totalNegs-10929 lines, as the first 35 and 10929 rows respectively are the test set...
df_test_ground= pd.read_csv(groundtruth, sep='\t', skipfooter=1826, engine='python') #!!!!!with DRUGBANK: 2783
df_test_neg_ground= pd.read_csv(neg_groundtruth, sep='\t', skipfooter=57670, engine='python') #!!!!!with DRUGBANK: 97228
test_neg_rtypes = df_test_neg_ground["RELATION"]
test_gener_refs = [0]*len(test_neg_rtypes)
df_test_neg_ground['REFERENCES'] = test_gener_refs
df_test = pd.concat([df_test_ground, df_test_neg_ground])
#dfLabels=["NOD1","NOD2","RELATION"]
df_test_pairs_rtypes= df_test.drop('REFERENCES', axis=1)
df_test_refs = df_test["REFERENCES"]

positivePairs = []
for index, row in df_test_ground.iterrows():
    positivePairs.append(str(row['NOD1'])+','+ str(row['NOD2']))
###keep ONLY positive test pairs for reference
del positivePairs[34:]

####undersampling original groundtruth ratio
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
#undersample = RandomUnderSampler(sampling_strategy=0.1)
#df_pairs_rtypes, refs = undersample.fit_resample(df_pairs_rtypes, refs)
###


FNs=0
TPs=0
FPs=0

VALIDATION_FSCORE10= []
VALIDATION_FSCORE5= []
VALIDATION_FSCORE15= []
VALIDATION_FSCORE20= []
VALIDATION_PRECISION = []
VALIDATION_RECALL= []
VALIDATION_FSCORE= []
VALIDATION_MRR= []
VALIDATION_HITS_AT1= []
VALIDATION_HITS_AT5= []
VALIDATION_HITS_AT10= []
VALIDATION_HITS_AT100= []



#    print('df_train_pairs len: ', len(df_train_pairs_rtypes))
#    print('df_train_pairs: ', df_train_pairs_rtypes)
#    print('df_train_ref len:', len (df_train_refs))
#    print('df_train_ref :', df_train_refs)
df_train=pd.concat([df_train_pairs_rtypes, df_train_refs], axis=1)
#    print('df_train len: ', len(df_train))
#    print('df_train: ', df_train)

    #undersampling training data, to set a different ratio
undersample = RandomUnderSampler(sampling_strategy=0.06)
df_train, df_train_refs = undersample.fit_resample(df_train, df_train_refs)
print('Negative, Positive train samples: ',Counter(df_train_refs))
#remove neg records from training...
df_neg_train = df_train[df_train['REFERENCES'] == 0]
df_train = df_train[df_train['REFERENCES'] > 0]

df_test=pd.concat([df_test_pairs_rtypes, df_test_refs], axis=1)
all_t_test = df_test["RELATION"]
src0 = [entity_type[index] for index in df_test["NOD1"]]
dst0 = [entity_type[index] for index in df_test["NOD2"]]
all_test_edge_index = torch.tensor([src0, dst0])
#######FOT! TRY to CREATE 2 DICTIONARIES - in order to check back dis/drug top pair predictions in groundtruth
#print("all_test_edge_index=", all_test_edge_index)
#print("size: ", all_test_edge_index.size())
dis_index_to_id = dict(zip(src0, df_test["NOD1"]))
drug_index_to_id = dict(zip(dst0, df_test["NOD2"]))
#print("drug_index_to_id: ", drug_index_to_id)

dis_id_to_index = dict(zip(df_test["NOD1"], src0))
drug_id_to_index = dict(zip(df_test["NOD2"], dst0))
######FOT!
all_test_edge_type = encodeEdgeTypes(all_t_test)
print("Negative, Positive test samples: " , Counter(df_test["REFERENCES"]))
df_pos_test = df_test[df_test['REFERENCES'] > 0]
df_neg_test = df_test[df_test['REFERENCES'] == 0] 



t_train = df_train["RELATION"]
pairLabels=["NOD1","NOD2"]
p_train = df_train[pairLabels]
a_train=df_train["REFERENCES"]
src = [entity_type[index] for index in p_train["NOD1"]]
dst = [entity_type[index] for index in p_train["NOD2"]]
train_edge_index = torch.tensor([src, dst])
train_edge_type = encodeEdgeTypes(t_train)
train_edge_attrs = [ie(a_train)]
train_edge_attr = torch.cat(train_edge_attrs, dim=-1)

####TRAIN WITH Groundtruth negative sample!
neg_p_train = df_neg_train[pairLabels]
nsrc = [entity_type[index] for index in neg_p_train["NOD1"]]
ndst = [entity_type[index] for index in neg_p_train["NOD2"]]
orig_neg_edge_index = torch.tensor([nsrc, ndst])
neg_t_train = df_neg_train["RELATION"]
orig_neg_edge_type = encodeEdgeTypes(neg_t_train)
####



##    p_train=pairs[train_index]
##    p_test=pairs[test_index]
##    t_train=rtypes[train_index]
##    t_test=rtypes[test_index]
##    a_train=refs[train_index]
##    a_test=refs[test_index]

t_test = df_pos_test["RELATION"]
p_test = df_pos_test[pairLabels]
a_test=df_pos_test["REFERENCES"]
src2 = [entity_type[index] for index in p_test["NOD1"]]
dst2 = [entity_type[index] for index in p_test["NOD2"]]
test_edge_index = torch.tensor([src2, dst2])
test_edge_type = encodeEdgeTypes(t_test)
test_edge_attrs = [ie(a_test)]
test_edge_attr = torch.cat(test_edge_attrs, dim=-1)

neg_t_test = df_neg_test["RELATION"]
neg_p_test = df_neg_test[pairLabels]
neg_a_test=df_neg_test["REFERENCES"]
src3 = [entity_type[index] for index in neg_p_test["NOD1"]]
dst3 = [entity_type[index] for index in neg_p_test["NOD2"]]
test_neg_edge_index = torch.tensor([src3, dst3])
test_neg_edge_type = encodeEdgeTypes(neg_t_test)
####

##FOR DEBUG PURPOSEs
#    print("pos t_test length=",len(t_test))
#    print("neg t_test length=",len(neg_t_test))
#    print("p_train[0]=",p_train.iloc[0] )
#    print("neg_p_train[0]=",neg_p_train.iloc[0] )
#    print ("train edge type: ", train_edge_type[0])
#    print("orig_neg_edge_type", orig_neg_edge_type[0])

prec=0.0
rec=0.0
f1=0.0
model = GAE(RGCNEncoder(data.num_nodes, hidden_channels=100, num_relations=rel_no),DistMultDecoder(rel_no // 2, hidden_channels=100),)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

####Training....
TP=0
FP=0
FN=0
for epoch in range(1, 191):  #INCREASE EPOCHS TO 191
    loss = train()
    if (epoch % 5) == 0:
        print(f'Epoch: {epoch:05d}, Loss: {loss:.4f}')
        prec, rec, f1, TP, FP, FN, mrr, hitsAt1, hitsAt5, hitsAt10, hitsAt100 = test()
        print('Epoch: ', epoch, '. Prec: ', prec, ' recall: ', rec, 'f1: ', f1)
    if epoch==5:
        VALIDATION_FSCORE5.append(f1)
    if epoch==10:
        VALIDATION_FSCORE10.append(f1)
    if epoch==15:
        VALIDATION_FSCORE15.append(f1)
    if epoch==20:
        VALIDATION_FSCORE20.append(f1)

VALIDATION_PRECISION.append(prec)
VALIDATION_RECALL.append(rec)
VALIDATION_FSCORE.append(f1)
VALIDATION_MRR.append(mrr)
VALIDATION_HITS_AT1.append(hitsAt1)
VALIDATION_HITS_AT5.append(hitsAt5)
VALIDATION_HITS_AT10.append(hitsAt10)
VALIDATION_HITS_AT100.append(hitsAt100)

TPs=TPs+TP
FPs=FPs+FP
FNs=FNs+FN
    
###################################################
##########Print marco-average of metric values#####
###################################################
from numpy import mean             
print("Average F1 after 5 epochs: ", mean(VALIDATION_FSCORE5))
print("Average F1 after 10 epochs: ", mean(VALIDATION_FSCORE10))
print("Average F1 after 15 epochs: ", mean(VALIDATION_FSCORE15))
print("Average F1 after 20 epochs: ", mean(VALIDATION_FSCORE20))

print("\nTest Set Metrics of the trained model:")
print("Average Precision: ", mean(VALIDATION_PRECISION))
print("Average Recall: ", mean(VALIDATION_RECALL))
print("Average F1-score: ", mean(VALIDATION_FSCORE))
print("Average MRR: ", mean(VALIDATION_MRR))
print("Average TAIL HITS@1: ", mean(VALIDATION_HITS_AT1))
print("Average TAIL HITS@5: ", mean(VALIDATION_HITS_AT5))
print("Average TAIL HITS@10: ", mean(VALIDATION_HITS_AT10))
print("Average TAIL HITS@100: ", mean(VALIDATION_HITS_AT100))

mPrec=TPs/(TPs+FPs)
mRec=TPs/(TPs+FNs)
print("micro-Average Precision: ", mPrec)
print("micro-Average Recall: ", mRec)
print("micro-Average F1-score: ", (2*mPrec*mRec/(mPrec+mRec)))
