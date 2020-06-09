# scrape.py
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import time
import progressbar
import logging

logging.basicConfig(filename='scrape.log', filemode='a+',
                    format='%(levelname)s - %(message)s')

def scrape(base_url='http://konect.uni-koblenz.de', vrange=(0, 20000),
            erange=(0, int(5e6)), categories=None, save_path='data/scraping/',
            wait_time=5, verbose=True):
    """
    Scrapes everything we can deal with from konect

    Parameters:
        vrange (tuple(int, int)): (min, max) vertex count to scrape
        erange (tuple(int, int)): (min, max) edge count to scrape
        categories (iterable(str)): iterable of strings specifying which
            categories to scrape from
    """
    if categories is None:
        # if no categories list unspecified, get everything
        categories = {'Affiliation', 'Animal', 'Authorship', 'Citation',
                      'Coauthorship', 'Communication', 'Computer', 'Feature',
                      'Folksonomy', 'HumanContact', 'HumanSocial', 'Hyperlink',
                      'Infrastructure', 'Interaction', 'Lexical', 'Metabolic',
                      'Misc', 'OnlineContact', 'Rating', 'Social', 'Software',
                      'Text', 'Trophic'}

    match_all = re.compile(r'.*')

    # list of list that will be fed into a df at the end
    temp_df = []

    source = requests.get(base_url + '/networks').text
    soup = BeautifulSoup(source, 'html.parser')

    table = soup.find('tbody')
    rows = table.findChildren(recursive=False)

    if verbose:
        rows = progressbar.progressbar(rows)

    for row in rows:
        # in case anything weird happens
        try:
            # check edge format for compatability
            edge_format = row.find('a', href='../help/network_format')
            edge_format = edge_format.find('img').get('alt')
            edge_format = edge_format.split(':')[0]

            # if the graph is not directed or undirected, skip it
            if edge_format == 'Bipartite':
                continue
            elif edge_format == 'Directed':
                directed = True
            else:
                directed = False

            # check weight format for compatability
            weight = row.find('a', href='../help/edge_weights').find('img').get('alt')
            weight = weight.split(':')[0]

            if weight in {'Multiple ratings', 'Ratings', 'Dynamic'}:
                continue
            elif weight in {'Unweighted', 'Multiple unweighted'}:
                weighted = False
            else:
                weighted = True

            # get category and weight information
            category = row.find('a', href='../help/categories').text[2:]

            if category not in categories:
                # check if we want this category
                continue

            # get number of vertices and edges
            v, e = row.find_all('td', style='text-align:right')[:2]
            v, e = v.text, e.text
            v, e = v.replace(',', ''), e.replace(',', '')
            vertices, edges = int(v), int(e)
            # check if we are in vertex/edge count limits
            if vertices < vrange[0] or vertices > vrange[1]:
                continue
            if edges < erange[0] or edges > erange[1]:
                continue

            # get the title and description
            title_tag = row.find(title=match_all)
            title = title_tag.text
            descr = title_tag.get('title')[15:]

            # get link to data
            link = row.find('a', href=re.compile(r'../downloads/tsv/.*')).get('href')
            data_link = base_url + link[2:]

            # get data
            # network_data = requests.get(data_link).content
            time.sleep(wait_time)   # give it a bit

            # save data
            temp_save = save_path + link.split('/')[-1]
            # with open(temp_save, 'wb') as f:
                # f.write(network_data)

            # append information to the dataframe
            network_info = [title, descr, category, directed, weighted,
                            vertices, edges, edge_format, weight, temp_save]
            temp_df.append(network_info)

        except KeyboardInterrupt:
            raise KeyboardInterrupt

        except Exception as e:
            logging.exception('exception occured')
            continue

    columns = ['name', 'description', 'category', 'directed', 'weighted',
               'vertices', 'edges', 'edge_format', 'weight_format', 'file_path']

    try:
        network_df = pd.DataFrame(temp_df, columns=columns)
        # network_df.sort_values('vertices', axis=1, inplace=True)
        network_df.to_csv(save_path + 'network_info.csv', sep='\t')

    except Exception as e:
        print(e)
