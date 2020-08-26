import pandas as pd 
from Bio import SeqIO
import numpy as np 

ds = pd.read_csv('protein_dataset.csv')

ds = ds[ds['sequence'].notna()]
ds = ds[ds['chain_id'].notna()]
ds = ds[ds['structure_id'].notna()]

chain = ds.iloc[:,1:2].values
seq = ds.iloc[:,5:6].values
pdb = ds.iloc[:,7:8].values

# print(chain, seq, pdb)

uniq_pdb = []

# f = open("protein_dataset.fasta", "w")

# for i in range(len(chain)):
# 	print(i, len(chain))
# 	try:
# 		pdbid = pdb[i][0]
# 		chainid = chain[i][0]
# 		seqstr = seq[i][0]
# 		# print(pdbid, chainid, seqstr)

# 		if pdbid not in uniq_pdb:
# 			string = '> ' + str(pdbid) + '_' + str(chainid)
# 			f.write(string)
# 			f.write("\n")
# 			f.write(seqstr)
# 			f.write("\n")
# 			uniq_pdb.append(pdbid)
# 	except:
# 		pass

# f.close()

pos_file = 'repr.1'

pdb_retain = []

for record in SeqIO.parse(pos_file, "fasta"):
    pdb_r = record.id.split('_')[0]
    print(pdb_r)
    pdb_retain.append(pdb_r)

print(len(pdb_retain))

keep = []

for i in range(len(pdb)):
	print(i)
	if pdb[i][0] in pdb_retain:
		keep.append(i)

ds = ds.iloc[keep]
print(ds)
ds.to_csv('protein_dataset_cdhit_80.csv')