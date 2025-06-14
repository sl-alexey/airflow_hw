import pandas as pd
import os
import dill
import json
import glob


from datetime import datetime

def predict():
    path = os.environ.get('PROJECT_PATH', '.')
    #path = os.path.expanduser('~/airflow_hw').replace('\\', '/')
    #path_to_models = f'{path}/data/models'
    #print(path_to_models)
    #print(os.listdir(path_to_models))

    with open(f'{path}/data/models/cars_pipe_202506140705.pkl', 'rb') as f:
        model = dill.load(f)
    df_pred = pd.DataFrame(columns=['car_id', 'pred'])
    for filename in glob.glob(f'{path}/data/test/*.json'):
        with open(filename) as fin:
            form = json.load(fin)
            df = pd.DataFrame.from_dict([form])
            y = model.predict(df)
            x = {'car_id': df.id, 'pred': y}
            df1 = pd.DataFrame(x)
            df_pred = pd.concat([df_pred, df1], axis=0)

    df_pred.to_csv(f'{path}/data/predictions/cars_pred_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)


if __name__ == '__main__':
    predict()
