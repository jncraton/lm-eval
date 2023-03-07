import os
import json
import pandas
import re

df = pandas.read_csv('open-models.csv')

df['normed-model'] = df['model'].str.replace('/','-')
df.set_index('normed-model', inplace=True)

for file in os.listdir('results'):
    if file.endswith('.json'):
        m = re.search("(.*?)\-([^\-]+\-?\d*)\.json", file)
        model = m.group(1)
        task = m.group(2)

        print(model, task)
    
        with open('results/' + file) as f:
            result = json.loads(f.read())

            acc = result['results'][0]['acc']

            df.at[model, task] = acc

df = df[['date', 'model', 'params', 'instruct', 'multi', 'boolq', 'copa', 'copa-1', 'copa-3', 'copa-5', 'cb', 'base', 'paper']]

df.to_csv('open-models-results.csv', index=False)
