import os
import pdb
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    headers = ["epoch","loss", "root_mean_squared_error", "ndcg_metric","recall_metric","hits_metric","mean_absolute_error","kl_div","val_loss","val_root_mean_squared_error","val_ndcg_metric","val_recall_metric","val_hits_metric","val_mean_absolute_error","val_kl_div"]

    with open('vasp_log.txt','r') as file:
        data = []
        epoch = 1
        for line in file:
            items = line.split(' ')
            if len(items) > 20:
                row = []
                row.append(epoch)
                if epoch <=50:
                    row.append(items[items.index('loss:') + 1])
                    row.append(items[items.index('root_mean_squared_error:') + 1])
                    row.append(items[items.index('ndcg_metric:') + 1])
                    row.append(items[items.index('recall_metric:') + 1])
                    row.append(items[items.index('hits_metric:') + 1])
                    row.append(items[items.index('mean_absolute_error:') + 1])
                    row.append(items[items.index('kl_div:') + 1])
                    row.append(items[items.index('val_loss:') + 1])
                    row.append(items[items.index('val_root_mean_squared_error:') + 1])
                    row.append(items[items.index('val_ndcg_metric:') + 1])
                    row.append(items[items.index('val_recall_metric:') + 1])
                    row.append(items[items.index('val_hits_metric:') + 1])
                    row.append(items[items.index('val_mean_absolute_error:') + 1])
                    row.append(items[items.index('val_kl_div:') + 1].split('\n')[0])
                elif epoch > 50 and epoch <=70:
                    row.append(items[items.index('loss:') + 1])
                    row.append(items[items.index('root_mean_squared_error:') + 1])
                    row.append(items[items.index('ndcg_metric_1:') + 1])
                    row.append(items[items.index('recall_metric_1:') + 1])
                    row.append(items[items.index('hits_metric_1:') + 1])
                    row.append(items[items.index('mean_absolute_error:') + 1])
                    row.append(items[items.index('kl_div:') + 1])
                    row.append(items[items.index('val_loss:') + 1])
                    row.append(items[items.index('val_root_mean_squared_error:') + 1])
                    row.append(items[items.index('val_ndcg_metric_1:') + 1])
                    row.append(items[items.index('val_recall_metric_1:') + 1])
                    row.append(items[items.index('val_hits_metric_1:') + 1])
                    row.append(items[items.index('val_mean_absolute_error:') + 1])
                    row.append(items[items.index('val_kl_div:') + 1].split('\n')[0])
                else:
                    row.append(items[items.index('loss:') + 1])
                    row.append(items[items.index('root_mean_squared_error:') + 1])
                    row.append(items[items.index('ndcg_metric_2:') + 1])
                    row.append(items[items.index('recall_metric_2:') + 1])
                    row.append(items[items.index('hits_metric_2:') + 1])
                    row.append(items[items.index('mean_absolute_error:') + 1])
                    row.append(items[items.index('kl_div:') + 1])
                    row.append(items[items.index('val_loss:') + 1])
                    row.append(items[items.index('val_root_mean_squared_error:') + 1])
                    row.append(items[items.index('val_ndcg_metric_2:') + 1])
                    row.append(items[items.index('val_recall_metric_2:') + 1])
                    row.append(items[items.index('val_hits_metric_2:') + 1])
                    row.append(items[items.index('val_mean_absolute_error:') + 1])
                    row.append(items[items.index('val_kl_div:') + 1].split('\n')[0])


                data.append(row)
                epoch += 1


            else:
                continue


        df = pd.DataFrame(data)
        df.columns = headers
        df.to_csv("data.csv")

        fig = plt.figure(figsize=(100, 100))
        plt.plot(df.epoch, df.loss,                     label = "loss")
        plt.plot(df.epoch, df.root_mean_squared_error,  label = "RMSE")
        plt.plot(df.epoch, df.ndcg_metric,              label = "NCDG")
        plt.plot(df.epoch, df.recall_metric,            label = "Recall")
        plt.plot(df.epoch, df.hits_metric,              label = "Hits")
        plt.plot(df.epoch, df.mean_absolute_error,      label = "MAE")
        plt.plot(df.epoch, df.kl_div,                   label = "kl_div")
        plt.legend()
        plt.savefig('training.png')

        fig = plt.figure(figsize=(100, 100))
        plt.plot(df.epoch, df.val_loss, label = "loss")
        plt.plot(df.epoch, df.val_root_mean_squared_error, label = "RMSE")
        plt.plot(df.epoch, df.val_ndcg_metric, label = "NCDG")
        plt.plot(df.epoch, df.val_recall_metric, label = "Recall")
        plt.plot(df.epoch, df.val_hits_metric, label = "Hits")
        plt.plot(df.epoch, df.val_mean_absolute_error, label = "MAE")
        plt.plot(df.epoch, df.val_kl_div, label = "kl_div")
        plt.legend()
        plt.savefig('validation.png')

if __name__ == '__main__':
    main()
