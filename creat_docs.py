module_name = 'xrspatial'
base_dir = '/home/yusin/docker_folder/RAG_bench/GeoCode/'
#DS1000-->
#ODEX--> all good
#cibench-->  all good
#BigCodeBench--->'soundfile'  xmltodict   wordninja   texttable
in_path = base_dir + module_name + '_signature.json'
#in_path = base_dir + module_name + '_doc.json'
out_path = base_dir + module_name

import json
import os
if not os.path.exists(out_path):
    os.mkdir(out_path)
# Load the original JSON file
with open(in_path, 'r') as infile:
    data = json.load(infile)

# Save each item as a separate JSON file
for key, value in data.items():
    output_file_path = os.path.join(out_path, f"{key}.json")
    with open(output_file_path, 'w') as outfile:
        json.dump('function:' + key + ' usage:' + value, outfile, indent=4)
