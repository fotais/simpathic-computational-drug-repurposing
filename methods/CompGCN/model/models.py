from helper import *
from model.compgcn_conv import CompGCNConv
from model.compgcn_conv_basis import CompGCNConvBasis
import csv
import ast
import pandas as pd 
import copy 


class BaseModel(torch.nn.Module):
	def __init__(self, params):
		super(BaseModel, self).__init__()
		self.p		= params[0]
		self.dict = params[1]



		self.similarity_tensor = torch.nn.Parameter(torch.Tensor(80, 80))

		df = pd.read_json('./data/{}/disease2disease_cuis.json'.format(self.p.dataset))
		df = df[sorted(df.columns)]
		column_names = df.columns.tolist()

		initial_tensor = torch.zeros(len(column_names), len(column_names))

		for index in range(len(column_names)):
			copied_list = copy.deepcopy(column_names)
			for value in df[column_names[index]]:
				for num in range(len(copied_list)):
					if value[1] == copied_list[num]:
						initial_tensor[index][num] = value[0]

		row_sums = initial_tensor.sum(dim=1, keepdim=True)
		row_sums = torch.where(row_sums > 0, row_sums, torch.ones_like(row_sums))
		initial_tensor = initial_tensor / row_sums

		with torch.no_grad():
			self.similarity_tensor.copy_(initial_tensor)

		self.act	= torch.tanh
		self.bceloss	= torch.nn.BCELoss()


	def loss(self, pred, true_label):
		return self.bceloss(pred, true_label)
		
class CompGCNBase(BaseModel):
	def __init__(self, edge_index, edge_type, num_rel, params=None):
		super(CompGCNBase, self).__init__(params)

		self.edge_index		= edge_index
		self.edge_type		= edge_type
		self.p.gcn_dim		= self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim
		self.init_embed		= get_param((self.p.num_ent,   self.p.init_dim))


		input = open('./data/{}/vectors.csv'.format(self.p.dataset))
		csvreader = csv.reader(input)
		count = 0
		dicc ={}
		for line in csvreader:
			dicc[line[0].lower()]=ast.literal_eval(line[1])
			count+=1


		extend  = [   torch.tensor (  dicc.get(self.dict[key], [0]*127)  ) for key in self.dict]
		stacked_extend = torch.nn.Parameter(torch.stack(extend, dim=0).float())


		self.init_embed =  stacked_extend

		input.close()




		self.device		= self.edge_index.device

		if self.p.num_bases > 0:
			self.init_rel  = get_param((self.p.num_bases,   127)) ###assumption 127 dimensions
		else:
			if self.p.score_func == 'transe': 	self.init_rel = get_param((num_rel,   127)) 
			else: 					self.init_rel = get_param((num_rel*2, 127))


		if self.p.num_bases > 0:
			self.conv1 = CompGCNConvBasis(127, self.p.gcn_dim, num_rel, self.p.num_bases, act=self.act, params=self.p)
			self.conv2 = CompGCNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer == 2 else None
		else:
			self.conv1 = CompGCNConv(127, self.p.gcn_dim,      num_rel, act=self.act, params=self.p)
			self.conv2 = CompGCNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer == 2 else None

		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

	def forward_base(self, sub, rel, drop1, drop2):

		r	= self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
		x, r	= self.conv1(self.init_embed, self.edge_index, self.edge_type, rel_embed=r)
		x	= drop1(x)
		x, r	= self.conv2(x, self.edge_index, self.edge_type, rel_embed=r) 	if self.p.gcn_layer == 2 else (x, r)
		x	= drop2(x) 							if self.p.gcn_layer == 2 else x
		
		df = pd.read_json('./data/{}/disease2disease_cuis.json'.format(self.p.dataset))

		df = df[sorted(df.columns)]
		column_names = df.columns.tolist()


		lowercase_column_names = [word.lower() for word in column_names]
		column_node_number = [key for key, value in self.dict.items() if value in lowercase_column_names]
		vecs = x[column_node_number]
		eye = torch.eye(self.similarity_tensor.size(0), device=self.similarity_tensor.device)
		sim_tensor = self.similarity_tensor * (1 - eye)
		new_vecs = torch.matmul(sim_tensor, vecs)
		mask = torch.zeros(x.size(0), 1, device=x.device)
		for idx in column_node_number:
			mask[idx] = 1.0

		new_embed = torch.zeros_like(x)
		for i, idx in enumerate(column_node_number):
			if i < len(new_vecs):  
				new_embed[idx] = (new_vecs[i]+x[idx])/2

		x = (mask * new_embed + (1 - mask) * x)		


		sub_emb	= torch.index_select(x, 0, sub)
		rel_emb	= torch.index_select(r, 0, rel)

		return sub_emb, rel_emb, x


class CompGCN_TransE(CompGCNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params[0].num_rel, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.drop, self.drop)
		obj_emb				= sub_emb + rel_emb

		x	= self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)		
		score	= torch.sigmoid(x)

		return score

class CompGCN_DistMult(CompGCNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params[0].num_rel, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.drop, self.drop)
		obj_emb				= sub_emb * rel_emb

		x = torch.mm(obj_emb, all_ent.transpose(1, 0))
		x += self.bias.expand_as(x)

		score = torch.sigmoid(x)
		return score

class CompGCN_ConvE(CompGCNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params[0].num_rel, params)

		self.bn0		= torch.nn.BatchNorm2d(1)
		self.bn1		= torch.nn.BatchNorm2d(self.p.num_filt)
		self.bn2		= torch.nn.BatchNorm1d(self.p.embed_dim)
		
		self.hidden_drop	= torch.nn.Dropout(self.p.hid_drop)
		self.hidden_drop2	= torch.nn.Dropout(self.p.hid_drop2)
		self.feature_drop	= torch.nn.Dropout(self.p.feat_drop)
		self.m_conv1		= torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz), stride=1, padding=0, bias=self.p.bias)

		flat_sz_h		= int(2*self.p.k_w) - self.p.ker_sz + 1
		flat_sz_w		= self.p.k_h 	    - self.p.ker_sz + 1
		self.flat_sz		= flat_sz_h*flat_sz_w*self.p.num_filt
		self.fc			= torch.nn.Linear(self.flat_sz, self.p.embed_dim)

	def concat(self, e1_embed, rel_embed):
		e1_embed	= e1_embed. view(-1, 1, self.p.embed_dim)
		rel_embed	= rel_embed.view(-1, 1, self.p.embed_dim)
		stack_inp	= torch.cat([e1_embed, rel_embed], 1)
		stack_inp	= torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.p.k_w, self.p.k_h))
		return stack_inp

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.hidden_drop, self.feature_drop)
		stk_inp				= self.concat(sub_emb, rel_emb)
		x				= self.bn0(stk_inp)
		x				= self.m_conv1(x)
		x				= self.bn1(x)
		x				= F.relu(x)
		x				= self.feature_drop(x)
		x				= x.view(-1, self.flat_sz)
		x				= self.fc(x)
		x				= self.hidden_drop2(x)
		x				= self.bn2(x)
		x				= F.relu(x)

		x = torch.mm(x, all_ent.transpose(1,0))
		x += self.bias.expand_as(x)

		score = torch.sigmoid(x)
		return score
