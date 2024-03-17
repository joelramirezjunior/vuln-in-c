import os 
import json 


DATASET_DIRECTORY = "./Dataset_1"
VULNERABILITY_MARKER = "VULNERABLE LINES"

#how to loop through all files in a directory. 



file_vul_map = {} 

for file in os.listdir(DATASET_DIRECTORY):

    with open(DATASET_DIRECTORY + "/" + file) as f:
        file_vul_map[file] = {}

        for line in reversed(list(f)):
            if VULNERABILITY_MARKER in line: 
                break
            if line == "\n": 
                continue
            vul_lines_index = line.strip().strip("//").strip().split(";")

            lines = [ x.split(",")[0] for x in vul_lines_index]
            indexes = [ x.split(",")[1] for x in vul_lines_index]
            file_vul_map[file][lines[0]] = indexes
        

for key, value in file_vul_map.items():
    print(key, "\n\t", value)


fp = open("map.json", "+w")
json.dump(file_vul_map, fp)

# dictionary that maps 

# file    ->  {                  
#             linenum -> [index1, index2],
#             linenum -> [index1, index2]
#             }   

# singhadityav08@gmail.com
