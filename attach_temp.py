import xml.etree.ElementTree as ET 
import csv
import pandas as pd
import multiprocessing

# d = {}
# for i in range(1, 86):
#     d[i] = pd.DataFrame(columns=['structure_id', 'chain_id', 'ec_no', 'source', 'classification', 'sequence', 'chain_length'])

# mapping = dict()
# mapping['dimEntity.structureId'] = 'structure_id'
# mapping['dimEntity.chainId'] = 'chain_id'
# mapping['dimEntity.ecNo'] = 'ec_no'
# mapping['dimEntity.source'] = 'source'
# mapping['dimStructure.classification'] = 'classification' 
# mapping['dimEntity.sequence'] = 'sequence'
# mapping['dimEntity.chainLength'] = 'chain_length'

# def make_df(filename, sno):
# 	try:
# 		tree = ET.parse(filename)
# 		root = tree.getroot()
# 		for record in list(root):
# 		    record_info = dict()
# 		    for attr in list(record):
# 		        record_info[mapping[attr.tag]]  = attr.text
# 		    d[sno] = d[sno].append(record_info, ignore_index=True)
# 		csv_name = filename + '.csv'
# 		d[sno].to_csv(csv_name)
# 		print("Done", i)
# 	except:
# 		pass

# jobs = []

# for i in range(1, 85):
# 	filename = "tmpfile_mp_" + str(i)
# 	print("Started", i)
# 	p = multiprocessing.Process(target=make_df, args=(filename, i))
# 	jobs.append(p)
# 	p.start()

# for proc in jobs:
#   proc.join()

protein_data = pd.DataFrame(columns=['structure_id', 'chain_id', 'ec_no', 'source', 'classification', 'sequence', 'chain_length'])

# print(d)

for i in range(1, 85):
	print(i)
	try:
		csv_name = filename = "temporary/tmpfile_mp_" + str(i) + '.csv'
		data = pd.read_csv(csv_name)
		protein_data = pd.concat([protein_data, data], axis = 0)
	except:
		pass

protein_data = protein_data.reset_index()
protein_data = protein_data.iloc[:,2:]

print(protein_data)
protein_data.to_csv("protein_dataset.csv")