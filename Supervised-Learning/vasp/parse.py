import pandas as pd
import json
import pdb

def parse():
    '''
    loss_df  = pd.DataFrame(columns=['epoch', 'loss', 'val_loss'])
    loss_f   = open('VASP_ML20_1_loss.json')
    data     = json.load(loss_f)
    epochs   = list(data['epochs'].values())
    loss     = list(data['loss'].values())
    val_loss = list(data['val_loss'].values())

    for e, lo, v_l in zip(epochs, loss, val_loss):
        loss_df = loss_df.append({'epoch':e, 'loss':lo, 'val_loss':v_l}, ignore_index= True)
    loss_df.to_csv('loss.csv')
    loss_f.close()
    '''

    metric_df  = pd.DataFrame(columns=['epochs', 'recall@5', 'recall@20', 'recall@50', 'ncdg@100', 'cov@5', 'cov@20', 'cov@50', 'cov@100'])
    metric_f   = open('VASP_ML20_1_metrics.json')
    data       = json.load(metric_f)
    epochs     = list(data['epochs'].values())
    recall_5   = list(data['Recall@5'].values())
    recall_20  = list(data['Recall@20'].values())
    recall_50  = list(data['Recall@50'].values())
    ncdg_100   = list(data['NCDG@100'].values())
    cov_5      = list(data['Coverage@5'].values())
    cov_20     = list(data['Coverage@20'].values())
    cov_50     = list(data['Coverage@50'].values())
    cov_100    = list(data['Coverage@100'].values())

    pdb.set_trace()
    for e, r_5, r_20, r_50, nc_100, c_5, c_20, c_50, c_100 in zip(epochs, recall_5, recall_20, recall_50, ncdg_100, cov_5, cov_20, cov_50, cov_100):
        metric_df = metric_df.append({'epochs':e, 'recall@5': r_5, 'recall@20': r_20, 'recall@50':r_50, 'ncdg@100':nc_100, 'cov@5' : c_5, 'cov@20':c_20, 'cov@50':c_50, 'cov@100':c_100}, ignore_index=True)

    metric_df.to_csv('metrics.csv')
    metric_f.close()




if __name__ == "__main__":
    parse()

