# mpi_makegraphs.py
from mpi4py import MPI
from load_txt import loadtxt
from sparse_specializer import DirectedGraph
from statistics import community_dist_bar
import pickle
import re
import os

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

regex = re.compile(r'out\..*')

base = '../'

directories = ['data/scraping/adjnoun_adjacency',
'data/scraping/sociopatterns-hypertext',
'data/scraping/foodweb-baywet',
'data/scraping/foodweb-baydry',
'data/scraping/radoslaw_email',
'data/scraping/maayan-foodweb',
'data/scraping/arenas-jazz',
'data/scraping/maayan-pdzbase',
'data/scraping/moreno_oz',
'data/scraping/moreno_innovation',
'data/scraping/contact',
'data/scraping/sociopatterns-infectious',
'data/scraping/arenas-meta',
'data/scraping/arenas-email',
'data/scraping/subelj_euroroad',
'data/scraping/moreno_blogs',
'data/scraping/maayan-faa',
'data/scraping/tntp-ChicagoRegional',
'data/scraping/opsahl-usairport',
'data/scraping/maayan-Stelzl',
'data/scraping/moreno_names',
'data/scraping/petster-friendships-hamster',
'data/scraping/moreno_propro',
'data/scraping/opsahl-ucsocial',
'data/scraping/dnc-temporalGraph',
'data/scraping/dnc-corecipient',
'data/scraping/maayan-figeys',
'data/scraping/petster-hamster',
'data/scraping/moreno_health',
'data/scraping/ego-facebook',
'data/scraping/opsahl-openflights',
'data/scraping/maayan-vidal',
'data/scraping/openflights',
'data/scraping/opsahl-powergrid',
'data/scraping/subelj_jung-j',
'data/scraping/reactome',
'data/scraping/subelj_jdk',
'data/scraping/as20000102',
'data/scraping/advogato',
'data/scraping/elec',
'data/scraping/chess',
'data/scraping/arenas-pgp',
'data/scraping/dblp-cite',
'data/scraping/foldoc',
'data/scraping/cfinder-google',
'data/scraping/ca-AstroPh',
'data/scraping/ca-cit-HepTh',
'data/scraping/eat',
'data/scraping/subelj_cora',
'data/scraping/ego-twitter',
'data/scraping/ego-gplus',
'data/scraping/as-caida20071105',
'data/scraping/cit-HepTh',
'data/scraping/ca-cit-HepPh',
'data/scraping/munmun_digg_reply',
'data/scraping/linux',
'data/scraping/cit-HepPh',
'data/scraping/topology']

directories = directories[RANK::SIZE]
#
# if RANK == 0:
#     directories = directories[0::8]
# elif RANK == 1:
#     directories = directories[1::8]
# elif RANK == 2:
#     directories = directories[2::8]
# elif RANK == 3:
#     directories = directories[3::8]
# elif RANK == 4:
#     directories = directories[4::8]
# elif RANK == 5:
#     directories = directories[5::8]
# elif RANK == 6:
#     directories = directories[6::8]
# elif RANK == 7:
#     directories = directories[7::8]

for graph in directories:
    try:
        for f in os.listdir(base+graph):
            if regex.match(f):
                fname = regex.match(f).string

        G = loadtxt(base+graph+'/'+fname)
        G = DirectedGraph(G)
        G.coloring()
        eq_part = G.colors
        # SAVE THE COLORING
        with open(f'../data/colorings/{fname}-coloring', 'wb') as f:
            pickle.dump(eq_part, f)

        community_dist_bar(eq_part, show=False, save=f'../data/graphs/{fname}.png')

    except KeyboardInterrupt as e:
        raise KeyboardInterrupt('keyboard interrupt')

    except Exception as e:
        print(e)
        continue
