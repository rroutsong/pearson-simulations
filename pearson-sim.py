import pandas as pd
import numpy as np
import time
from pprint import pprint as pp
import plotly.plotly as py
import plotly.graph_objs as go

full_matrix = pd.read_csv('merged_results.tsv', sep='\t')

pearson_results = []
compute_times = {}

for x in range(20, 220, 2):
    start = time.time()
    subset_matrix = full_matrix.iloc[0:x+20]

    geneids = subset_matrix['geneid'].tolist()  # gene ids from query list
    query_values = subset_matrix.iloc[:, 1]  # query values for pearson correlation

    match = pd.DataFrame(columns=full_matrix.columns)

    for i in range(0, len(geneids)):
        match = match.append(full_matrix.loc[full_matrix['geneid'] == geneids[i]])

    for i in list(match.columns)[1:]:
        with np.errstate(divide='ignore'):
            log2q = np.log2(query_values)
            log2q[np.isneginf(log2q)] = 0
            log2m = np.log2(match[i])
            log2m[np.isneginf(log2m)] = 0
        x_pearson = [x, query_values.name, match[i].name,
                     query_values.corr(match[i], method='pearson'),
                     log2q.corr(log2m, method='pearson')]
        pearson_results.append(x_pearson)

    end = time.time()
    compute_time = (end-start)

    if x in compute_times.keys():
        compute_times[x].append(compute_time)
    else:
        compute_times[x] = [compute_time]

pearson_df = pd.DataFrame(pearson_results, columns=['gene count', 'query name', 'match name', 'raw R2', 'log2 R2'])

trace1 = go.Scatter(
    y=pearson_df['gene count'].tolist(),
    x=pearson_df['raw R2'].tolist(),
    mode='markers',
    name='Raw R',
    marker= dict(
        color='rgb(66, 134, 244)'
    )
)

trace2 = go.Scatter(
    y=pearson_df['gene count'].tolist(),
    x=pearson_df['log2 R2'].tolist(),
    mode='markers',
    name='Log2 R',
    marker=dict(
        color='rgb(244, 134, 65)'
    )
)

py.plot([trace1, trace2], filename='basic-scatter', auto_open=True)
