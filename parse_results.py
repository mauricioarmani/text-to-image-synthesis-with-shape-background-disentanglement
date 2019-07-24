import json
from glob import glob


jsonfiles = glob('results/*.json')
jsonfiles.sort()

for jsonfile in jsonfiles:
    results = json.load(open(jsonfile))
    for key, value in results.items():
        if key == 'mean' or key == 'std':
            print(jsonfile, key, value['output_64'])
        else:
            print(jsonfile, key, value)