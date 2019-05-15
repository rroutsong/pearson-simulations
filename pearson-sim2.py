import pandas as pd
import random
from scipy.stats import zscore,pearsonr
import numpy as np
import time
from pprint import pprint as pp
import plotly.plotly as py
import plotly.graph_objs as go

full_matrix = pd.read_csv('data/merged.txt', sep='\t')

geneids = full_matrix['geneid'].tolist()

pearson_results = []
compute_times = {}


def neglog10(x):
    return -1*np.log10(x)


for x in range(20, 2000, 5):
    start = time.time()
    selected_geneids = random.sample(geneids, x)

    match = pd.DataFrame(columns=full_matrix.columns)

    for i in range(0, len(selected_geneids)):
        match = match.append(full_matrix.loc[full_matrix['geneid'] == selected_geneids[i]])

    query = match.iloc[:, :2]
    q_geneid = query.pop('geneid')
    query_z = query.apply(zscore)
    query_z.insert(0, 'geneid', q_geneid)

    m_geneid = match.pop('geneid')
    match_z = match.apply(zscore)
    match_z.insert(0, 'geneid', m_geneid)

    for i in list(match_z.columns)[1:]:
        r = pearsonr(match_z[i], query_z.iloc[:,1])
        x_pearson = [x, query_z.iloc[:,1].name, match_z[i].name, r[0], r[1]]
        pearson_results.append(x_pearson)

    end = time.time()
    compute_time = (end - start)

    if x in compute_times.keys():
        compute_times[x].append(compute_time)
    else:
        compute_times[x] = [compute_time]

pearson_df = pd.DataFrame(pearson_results, columns=['gene count', 'query name', 'match name', 'R', 'p-value'])

trace1 = go.Scatter(
    y=pearson_df['gene count'].tolist(),
    x=pearson_df['R'].tolist(),
    mode='markers',
    name='R',
    marker= dict(
        color='rgb(66, 134, 244)'
    )
)

trace2 = go.Scatter(
    y=pearson_df['gene count'].tolist(),
    x=pearson_df['p-value'].apply(neglog10).tolist(),
    mode='markers',
    name='p-value',
    marker=dict(
        color='rgb(244, 134, 65)'
    )
)

layout = go.Layout(
    title=go.layout.Title(
        text='R Plot',
        xref='paper',
        x=0
    ),
    xaxis=go.layout.XAxis(
        title='R'
    ),
    yaxis=go.layout.YAxis(
        title='Genes'
    )
)

fig1 = go.Figure(data=[trace1], layout=layout)

py.plot(fig1, filename='R distribution', auto_open=True)

barlayout = go.Layout(
    title=go.layout.Title(
        text='Compute time',
        xref='paper',
        x=0
    ),
    xaxis=go.layout.XAxis(
        title='Genes'
    ),
    yaxis=go.layout.YAxis(
        title='Compute time'
    )
)

bardata = go.Scatter(
            x=list(compute_times.keys()),
            y=list(compute_times.values()),
            mode='lines+markers'
          )

fig2 = go.Figure(data=[bardata], layout=barlayout)

py.plot(fig2, filename='R distribution compute time', auto_open=True)

pval = pearson_df['p-value'].tolist()
pval.sort(reverse=True)
