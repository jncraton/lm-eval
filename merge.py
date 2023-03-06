import os
import json
import pandas

df = pandas.read_csv('open-models.csv')

df['normed-model'] = df['model'].str.replace('/','-')
df.set_index('normed-model', inplace=True)

for file in os.listdir('results'):
    if file.endswith('.json'):
        model = '-'.join(file.split('-')[:-1])
    
        with open('results/' + file) as f:
            result = json.loads(f.read())

            task = result['results'][0]['task_name']
            acc = result['results'][0]['acc']
            
            df.at[model, task] = acc

df = df[['date', 'model', 'params', 'instruct', 'multi', 'boolq', 'copa', 'cb', 'base', 'paper']]

df.to_csv('open-models-results.csv', index=False)
