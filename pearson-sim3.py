import pandas as pd
import random
from scipy.stats import zscore,pearsonr
import numpy as np
import time
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import scipy.spatial.distance as dist
from pprint import pprint as pp
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff

full_matrix = pd.read_csv('data/poplardb.txt', sep='\t')

pearson_results = []

input_gene_list = """Potri.018G131100
Potri.013G131200
Potri.010G156300
Potri.013G040000
Potri.017G051700
Potri.003G084000
Potri.002G013000
Potri.T131700
Potri.008G086700
Potri.014G120400
Potri.005G011400
Potri.009G030200
Potri.014G034700
Potri.010G055300
Potri.015G004700
Potri.006G002300
Potri.006G036200
Potri.002G196800
Potri.013G025800
Potri.004G012100
Potri.019G014700
Potri.012G019200
Potri.007G068500
Potri.013G028800
Potri.001G450000
Potri.006G037800
Potri.011G049900
Potri.015G131300
Potri.013G102900
Potri.011G026300
Potri.004G146000
Potri.T104400
Potri.002G234200
Potri.001G147300
Potri.008G036900
Potri.005G084100
Potri.004G032900
Potri.005G233900
Potri.005G180200
Potri.014G193800
Potri.005G239000
Potri.001G065900
Potri.011G058000
Potri.014G154200
Potri.008G132700
Potri.013G156100
Potri.003G171800
Potri.001G237600
Potri.010G032800
Potri.001G061100
Potri.007G042400
Potri.010G208800
Potri.014G096900
Potri.001G318000
Potri.009G085600
Potri.010G083600
Potri.014G044100
Potri.017G153200
Potri.002G081000
Potri.016G052300
Potri.002G234000
Potri.001G355100
Potri.002G224100
Potri.011G080400
Potri.008G073800
Potri.008G175000
Potri.005G057400
Potri.017G140900
Potri.005G235200
Potri.003G143300
Potri.004G155500
Potri.014G106600
Potri.004G215400
Potri.006G237200
Potri.017G059800
Potri.016G004400
Potri.010G027600
Potri.012G004500
Potri.010G156500
Potri.001G301800
Potri.008G066200
Potri.015G056100
Potri.001G454900
Potri.003G107100
Potri.003G062400
Potri.008G079100
Potri.002G188500
Potri.004G149200
Potri.009G097800
Potri.009G034500
Potri.001G047500
Potri.008G174900
Potri.007G013100
Potri.011G151600
Potri.004G204700
Potri.007G018000
Potri.005G024800
Potri.009G133100
Potri.005G040700
Potri.010G061100
Potri.016G024500
Potri.002G234100
Potri.002G072000
Potri.017G014200
Potri.011G044700
Potri.007G001200
Potri.016G132600
Potri.015G048700
Potri.005G246800
Potri.013G040100
Potri.001G209300
Potri.006G024200
Potri.014G154300
Potri.007G096200
Potri.009G143700
Potri.018G044100
Potri.004G033000
Potri.013G103000
Potri.004G184000
Potri.006G129900
Potri.008G159000
Potri.009G063100
Potri.002G018000
Potri.010G061700
Potri.009G062800
Potri.014G006900
Potri.002G203500
Potri.T149500
Potri.013G102700
Potri.001G437400
Potri.016G102100
Potri.008G070200
Potri.001G455400
Potri.013G083500
Potri.001G123800
Potri.010G193000
Potri.017G150100
Potri.006G136300
Potri.008G036300
Potri.001G232200
Potri.003G197800
Potri.004G086600
Potri.009G038300
Potri.015G087500
Potri.013G086600
Potri.005G249700
Potri.008G179400
Potri.006G200400
Potri.001G231600
Potri.011G044500
Potri.001G279000
Potri.004G196000
Potri.011G011200
Potri.010G039300
Potri.001G213300
Potri.006G137000
Potri.005G141900
Potri.003G159600
Potri.011G122900
Potri.002G186400
Potri.002G065200
Potri.006G056200
Potri.014G197500
Potri.015G109600
Potri.011G067800
Potri.009G072900
Potri.002G028800
Potri.018G115400
Potri.012G091100
Potri.012G139400
Potri.004G216700
Potri.015G110400
Potri.015G083200
Potri.008G198600
Potri.001G213500
Potri.008G069100
Potri.015G132500
Potri.015G013300
Potri.005G093200
Potri.011G032600
Potri.005G115000
Potri.005G200400
Potri.011G053600
Potri.004G162600
Potri.001G128100
Potri.005G218900
Potri.006G124100
Potri.018G130100
Potri.018G130000
Potri.014G173600
Potri.008G092700
Potri.005G043400
Potri.005G158100
Potri.004G037800
Potri.002G048600
Potri.014G161300
Potri.010G111600
Potri.015G077600
Potri.006G153300
Potri.012G021700
Potri.015G087400
Potri.002G099300
Potri.015G000500
Potri.006G199100
Potri.001G365500
Potri.004G161400
Potri.004G174400
Potri.007G011200
Potri.013G151500
Potri.013G114800
Potri.017G039400
Potri.009G133600
Potri.T093500
Potri.001G376200
Potri.005G236500
Potri.015G143700
Potri.010G239300
Potri.001G364000
Potri.002G011800
Potri.013G014200
Potri.004G165300
Potri.004G222800
Potri.004G210500
Potri.006G026500
Potri.004G135300
Potri.003G143200
Potri.001G270800
Potri.002G155600
Potri.017G097000
Potri.001G268500
Potri.008G175500
Potri.010G188200
Potri.014G018400
Potri.004G235500
Potri.008G162800
Potri.004G099500
Potri.006G133500
Potri.017G020000
Potri.001G170500
Potri.016G023200
Potri.002G091600
Potri.010G038200
Potri.002G260000
Potri.T035400
Potri.012G039100
Potri.012G141500
Potri.010G150800
Potri.T051300
Potri.009G112500
Potri.016G138600
Potri.008G113400
Potri.016G077000
Potri.009G037500
Potri.009G042600
Potri.016G051900
Potri.002G231900
Potri.008G076100
Potri.007G086000
Potri.005G071900
Potri.010G181200
Potri.004G074000
Potri.002G143100
Potri.018G022100
Potri.002G103300
Potri.016G031400
Potri.001G449600
Potri.004G233800
Potri.012G128100
Potri.007G008700
Potri.012G101400
Potri.008G151000
Potri.015G042800
Potri.003G056200
Potri.017G026000
Potri.008G088300
Potri.006G263500
Potri.014G109200
Potri.003G102800
Potri.004G080000
Potri.002G103400
Potri.004G135200
Potri.012G128600
Potri.001G071000
Potri.013G031100
Potri.006G188500
Potri.006G106900
Potri.006G226700
Potri.001G013400
Potri.004G212900
Potri.003G146400
Potri.002G251200
Potri.019G069300
Potri.018G112300
Potri.010G208300
Potri.016G051500
Potri.010G167200
Potri.014G127300
Potri.015G039000
Potri.013G027900
Potri.001G457300
Potri.017G111900
Potri.009G110800
Potri.017G146700
Potri.016G132200
Potri.013G093700
Potri.010G210300
Potri.005G092500
Potri.016G083500
Potri.001G332200
Potri.014G019600
Potri.005G195600
Potri.003G222700
Potri.013G015300
Potri.015G094400
Potri.006G195400
Potri.015G007100
Potri.003G020200
Potri.016G134600
Potri.010G234100
Potri.017G075500
Potri.016G098200
Potri.015G087800
Potri.016G077200
Potri.010G207600
Potri.006G033500
Potri.018G090300
Potri.014G096100
Potri.014G147700
Potri.015G084700
Potri.010G066400
Potri.011G046600
Potri.003G050900
Potri.004G015300
Potri.005G044400
Potri.007G071800
Potri.007G131800
Potri.006G186200
Potri.003G161200
Potri.003G085100
Potri.017G036800
Potri.011G118900
Potri.010G186500
Potri.013G007900
Potri.006G126800
Potri.001G036900
Potri.017G083000"""

q_gene_list = input_gene_list.split('\n')

x4cl_full = pd.DataFrame()
x4cl_full['gene'] = full_matrix['gene']
x4cl_full['x4cl'] = full_matrix['X4cl']

x4cl_query = pd.DataFrame(columns=x4cl_full.columns)

for i in range(0, len(q_gene_list)):
    x4cl_query = x4cl_query.append(x4cl_full.loc[x4cl_full['gene'] == q_gene_list[i]])

def neglog10(x):
    return -1*np.log10(x)


start = time.time()

match = pd.DataFrame(columns=full_matrix.columns)

for i in range(0, len(q_gene_list)):
    match = match.append(full_matrix.loc[full_matrix['gene'] == q_gene_list[i]])

print(f'Shape of original matrix: {match.iloc[:,1:].shape}')

value_matrix = match.iloc[:,1:].values.tolist()
dist_matrix = dist.pdist(value_matrix)
sq_dist_matrix = dist.squareform(dist_matrix)
linkage_matrix = linkage(sq_dist_matrix)



# Dendrogram plot
#
# dendro = ff.create_dendrogram(linkage_matrix)
# py.plot(dendro)

exit()

# Z-Scores

#q_geneid = query.pop('geneid')
#query_z = query.apply(zscore)
#query_z.insert(0, 'geneid', q_geneid)

#m_geneid = match.pop('geneid')
#match_z = match.apply(zscore)
#match_z.insert(0, 'geneid', m_geneid)

for i in list(match.columns)[1:]:
    r = pearsonr(match[i], x4cl_query.iloc[:,1])
    x_pearson = [x4cl_query.iloc[:,1].name, match[i].name, r[0], r[1]]
    pearson_results.append(x_pearson)

end = time.time()
compute_time = (end - start)

pearson_df = pd.DataFrame(pearson_results, columns=['query name', 'match name', 'R', 'p-value'])

pp(pearson_df)

pearson_df.to_csv('poplardb_query.txt', sep='\t')
