import requests 
import json
import xml.etree.ElementTree as ET 
import csv
import pandas as pd

URL = "http://search.rcsb.org/rcsbsearch/v1/query"

PARAMS = {
  "query": {
    "type": "terminal",
    "service": "text",
    "parameters": {
      "operator": "less_or_equal",
      "value": "2020-08-15T00:00:00Z",
      "attribute": "rcsb_accession_info.initial_release_date"
    }
  },
  "request_options": {
      "pager": {
          "start": 0,
          "rows": 2000
      }
  },
  "return_type": "entry"
}

api_calls = 1
all_identifiers = []
curr_row = 0

while True:
    print("Get all identifiers: ", api_calls)
    r = requests.post(URL, json=PARAMS)
    api_calls += 1
    data = r.json()
    curr_row += 2000
    for element in data["result_set"]:
        all_identifiers.append(element["identifier"])
    total_count = data["total_count"]
    if curr_row > total_count:
        break
    else:
        PARAMS["request_options"]["pager"]["start"] = curr_row

CUSTOM_REPORT_URL = "http://www.rcsb.org/pdb/rest/customReport"
CUSTOM_REPORT_PARAMS = {
  "pdbids": "4LVN",
  "customReportColumns": "ecNo,source,classification,sequence,chainLength"
}

api_calls = 1
start = 0
step = 2000
i = 0
protein_data = pd.DataFrame(columns=['structure_id', 'chain_id', 'ec_no', 'source', 'classification', 'sequence', 'chain_length'])

mapping = dict()
mapping['dimEntity.structureId'] = 'structure_id'
mapping['dimEntity.chainId'] = 'chain_id'
mapping['dimEntity.ecNo'] = 'ec_no'
mapping['dimEntity.source'] = 'source'
mapping['dimStructure.classification'] = 'classification' 
mapping['dimEntity.sequence'] = 'sequence'
mapping['dimEntity.chainLength'] = 'chain_length'

while True:
    ids_to_send = all_identifiers[start]
    while i < start + step and i < len(all_identifiers):        
        ids_to_send += ","
        ids_to_send += all_identifiers[i]
        i += 1
    if ids_to_send[-1] == ',':
        ids_to_send = ids_to_send[:-1]
    CUSTOM_REPORT_PARAMS["pdbids"] = ids_to_send    
    result = requests.get(CUSTOM_REPORT_URL, CUSTOM_REPORT_PARAMS)
    print("API Call #", api_calls, "to fetch results starting @", start)
    api_calls += 1
    with open("tmpfile", "wb") as writefile:
        writefile.write(result.content)
    tree = ET.parse("tmpfile")
    root = tree.getroot()
    for record in list(root):
        record_info = dict()
        for attr in list(record):
            record_info[mapping[attr.tag]]  = attr.text
        protein_data = protein_data.append(record_info, ignore_index=True)
    start += step
    if start >= len(all_identifiers):
        break
protein_data.to_csv("protein_dataset.csv")