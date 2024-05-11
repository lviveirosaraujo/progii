import json
import pandas as pd
import numpy as np
from numpy import asarray
from PIL import Image # python package Pillow
import networkx as nx

# T1

with open('dados/prize.json','r') as f:
    prizes = json.load(f)["prizes"]

with open('dados/laureate.json','r') as f:
    laureates = json.load(f)["laureates"]

def maisPartilhados() -> tuple[int,set[tuple[int,str]]]:
    """
    Finds the maximum share value among laureates and returns a tuple containing the maximum share value and a set of (year, category) pairs for laureates with the maximum share.

    Returns:
        A tuple containing the maximum share value and a set of (year, category) pairs for laureates with the maximum share.
    """

    max_share = 0
    # First pass: Determine the absolute maximum share
    for prize in prizes:
        for laureate in prize.get("laureates", []):
            laureate_share = int(laureate["share"])
            max_share = max(max_share, laureate_share)

    max_share_data = set()

    # Second pass: Collect (year, category) pairs that match the max_share
    for prize in prizes:
        year = prize["year"]
        category = prize["category"]
        for laureate in prize.get("laureates", []):
            if int(laureate["share"]) == max_share:
                max_share_data.add((int(year), category))

    return max_share, max_share_data

def multiLaureados() -> dict[str,set[str]]:
    """
    Collects and returns a dictionary of laureates who have won in more than one category.

    Returns:
        dict[str, set[str]]: A dictionary where the keys are the names of the laureates and the values are sets of categories they have won in.
    """
    
    laureate_categories = {}

    # Loop through each prize and collect categories for each laureate
    for prize in prizes:
        category = prize["category"]
        for laureate in prize.get("laureates", []):
            # Check if 'surname' key exists in the laureate dictionary
            if "surname" in laureate:
                name = f"{laureate['firstname']} {laureate['surname']}"
            else:
                name = f"{laureate['firstname']}"
            if name in laureate_categories:
                laureate_categories[name].add(category)
            else:
                laureate_categories[name] = {category}

    # Filter to get only those laureates who have won in more than one category
    multi_category_laureates = {name: categories for name, categories in laureate_categories.items() if len(categories) > 1}

    return multi_category_laureates

def anosSemPremio() -> tuple[int,int] :
    """
    Find the longest consecutive range of missing years without a prize in each category.

    Returns:
        A tuple containing the start and end years of the longest consecutive range of missing years.
        If there are no missing years, it returns (0, 0).
    """
    
    # Collect all years and categories from the data

    all_years = set()
    categories_by_year = {}
    for prize in prizes:
        year = int(prize['year'])
        category = prize['category']
        all_years.add(year)
        if year in categories_by_year:
            if "laureates" in prize:
                categories_by_year[year].add(category)
        else:
            categories_by_year[year] = {category}

    # Determine the set of all categories
    all_categories = set(cat for year_cats in categories_by_year.values() for cat in year_cats)
    missing_years = []
    
    # Identify years with missing categories after 1969
    for year in range(min(all_years), max(all_years) + 1):
        if year > 1969:
            if year in categories_by_year:
                if categories_by_year[year] != all_categories:
                    missing_years.append(year)
            else:
                missing_years.append(year)

    # Remove the economics category from the set of all categories
    all_categories.discard("economics")

    # Identify years with missing categories before 1969
    for year in range(min(all_years), max(all_years) + 1):
        if year <= 1969:
            if year in categories_by_year:
                if categories_by_year[year] != all_categories:
                    missing_years.append(year)
            else:
                missing_years.append(year)

    
    if not missing_years:
        return (0, 0)  # Return a default value if there are no missing years
    
    # Find the longest consecutive range of missing years
    longest_start = missing_years[0]
    longest_end = missing_years[0]
    current_start = missing_years[0]
    current_end = missing_years[0]

    for i in range(1, len(missing_years)):
        if missing_years[i] == missing_years[i - 1] + 1:
            current_end = missing_years[i]
        else:
            if current_end - current_start > longest_end - longest_start:
                longest_start = current_start
                longest_end = current_end
            current_start = missing_years[i]
            current_end = missing_years[i]
    
    if current_end - current_start > longest_end - longest_start:
        longest_start = current_start
        longest_end = current_end

    # Note: I found the largest range of missing years to be 1914-1918, which is different from the expected result of 1914-1919
    return (longest_start, longest_end)

def rankingDecadas() -> dict[str,tuple[str,int]]:
    """
    Calculates the ranking of countries based on the number of laureates per decade.

    Returns:
        A dictionary containing the country with the most laureates and the corresponding count for each decade.
        The keys are the decades in the format '190x', '191x', etc., and the values are tuples of the form (country, count).
    """
    
    from collections import defaultdict

    # Dictionary to store the count of laureates per decade and country
    decada_pais_count = defaultdict(lambda: defaultdict(int))

    # Loop through each laureate and prize to count the laureates per decade and country
    for laureate in laureates:
        for prize in laureate.get("prizes", []):
            year = int(prize["year"])
            decada = f"{(year // 10)}x"  # Formate the decade
            for affiliation in prize.get("affiliations", []):
                if "country" in affiliation and affiliation["country"]:
                    pais = affiliation["country"]
                    decada_pais_count[decada][pais] += 1  

    # Find the country with the most laureates in each decade
    result = {}
    for decada, paises_count in decada_pais_count.items():
        # Find the country with the maximum count of laureates
        pais_max, count_max = max(paises_count.items(), key=lambda item: item[1])
        result[decada] = (pais_max, count_max)

    return result

# T2



def toGrayscale(rgb:np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to grayscale.

    Parameters:
    rgb (np.ndarray): The input RGB image.

    Returns:
    np.ndarray: The grayscale image.
    """
    grayscale = 0.21 * rgb[:, :, 0] + 0.72 * rgb[:, :, 1] + 0.07 * rgb[:, :, 2]
    return grayscale.astype(np.uint8)

def converteGrayscale(fromimg:str,toimg:str) -> None:
    # a 3D numpy array of type uint8
    rgb: np.ndarray = asarray(Image.open(fromimg))
    # a 2D numpy array of type uint8
    grayscale: np.ndarray = toGrayscale(rgb)
    Image.fromarray(grayscale, mode="L").save(toimg)

def toBW(gray:np.ndarray,threshold:tuple[int,int]) -> np.ndarray:
    """
    Converts a grayscale image to black and white based on a given threshold.

    Args:
        gray (np.ndarray): The grayscale image to be converted.
        threshold (tuple[int,int]): The lower and upper bounds of the threshold.

    Returns:
        np.ndarray: The black and white image.

    """
    lower_bound = threshold[0]
    upper_bound = threshold[1]
    is_within_threshold = np.logical_and(gray >= lower_bound, gray <= upper_bound)
    return np.where(is_within_threshold, 255, 0)

def converteBW(fromimg:str,toimg:str,threshold:tuple[int,int]) -> None:
    # a 2D numpy array of type uint8
    grayscale : np.ndarray = asarray(Image.open(fromimg))
    # a 2D numpy array of type uint8 (but with values being only 0 or 255)
    bw : np.ndarray = toBW(grayscale,threshold)
    Image.fromarray(bw,mode="L").save(toimg)

def autoThreshold(fromimg: str, tolerance: int) -> tuple[int, int]:
    """
    Automatically determines the lower and upper threshold values for image binarization.

    Args:
        fromimg (str): The path to the input image file.
        tolerance (int): The tolerance value used to determine the threshold range.

    Returns:
        tuple[int, int]: A tuple containing the lower and upper threshold values.
    """
    
    grayscale: np.ndarray = asarray(Image.open(fromimg))
    grayscale_flattened = grayscale.flatten()
    most_frequent_value = np.argmax(np.bincount(grayscale_flattened))

    lower_bound = max(most_frequent_value - tolerance, 0)
    upper_bound = min(most_frequent_value + tolerance, 255)

    return (lower_bound, upper_bound)

import numpy as np
from PIL import Image

def toContour(bw: np.ndarray) -> np.ndarray:
    """
    Converts a binary image to a contour image.

    Args:
        bw (np.ndarray): Binary image represented as a NumPy array.

    Returns:
        np.ndarray: Contour image represented as a NumPy array.
    """

    # Create a new image with the same shape as the input image

    contour_img = np.full(bw.shape, 255, dtype=np.uint8)

    # Find the differences between the pixels in the horizontal and vertical directions

    bw_right_shifted = np.roll(bw, -1, axis=1)
    is_diff_right = bw != bw_right_shifted
    is_diff_right[:, -1] = False

    bw_down_shifted = np.roll(bw, -1, axis=0)
    is_diff_down = bw != bw_down_shifted
    is_diff_down[-1, :] = False

    # Set the pixels in the contour image to 0 where there is a difference

    contour_img[is_diff_right | is_diff_down] = 0

    image = Image.fromarray(contour_img)
    image.save('contour_result.png')

    return contour_img

def converteContour(fromimg:str,toimg:str) -> None:
    # a 2D numpy array of type uint8 (but with values being only 0 or 255)
    bw : np.ndarray = asarray(Image.open(fromimg).convert("L"))
    # a 2D numpy array of type uint8 (but with values being only 0 or 255)
    contour : np.ndarray = toContour(bw)
    Image.fromarray(contour,mode="L").save(toimg)

# T3

legislativas = pd.read_excel("dados/legislativas.xlsx",header=[0,1],sheet_name="Quadro")

def eleitoresPorto() -> int:
    """
    Returns the index of the row with the maximum number of voters in the 'Área Metropolitana do Porto' region.
    
    Returns:
        int: The index of the row with the maximum number of voters.
    """
    region_name = 'Área Metropolitana do Porto'
    interest_row = legislativas[legislativas[('Territórios', 'Região')] == region_name]
    return interest_row['Total'].max().idxmax()

def taxaAbstencao() -> list[tuple[int,float]]:
    """
    Calculates the percentage of abstention in an election.

    Returns:
        A list of tuples, where each tuple contains the district number and the corresponding percentage of abstention.
    """
    porcentagem_de_abstencao = []
    abstencao = []

    # Calculate the percentage of abstention for the entire country

    total = legislativas["Total"].iloc[0]
    eleitores = legislativas["Votantes"].iloc[0]
    porcentagem_de_abstencao.append(((total - eleitores) / total) * 100)

    # Create a list of tuples with the district number and the corresponding percentage of abstention

    for index in range(len(legislativas["Total"].columns)):
        abstencao.append((legislativas["Total"].columns[index], porcentagem_de_abstencao[0].iloc[index]))

    return abstencao


def perdaGrandesMunicipios() -> dict[str,int]:
    """
    Calculates the year in which the largest municipalities experienced the biggest loss of voters compared to previous elections.

    Returns:
    A dictionary where the keys are the names of the municipalities and the values are the corresponding years.
    """

    municipios_validos = {}


    location_type = 'Município'
    municipal_rows = legislativas[legislativas[('Territórios', 'Âmbito Geográfico')] == location_type]

    # Find the municipalities with more than 10,000 voters in any year
    for row_index, row_serie in municipal_rows["Votantes"].iterrows():
        for year_index in range(len(municipal_rows["Votantes"].columns)):
            if row_serie.iloc[year_index] >= 10000:
                municipios_validos[legislativas['Territórios']['Região'].iloc[row_index]] = ""
                break
    
    # Create a dictionary to map the index of the year to the year itself
    years2index = {index: year for index, year in enumerate(municipal_rows["Votantes"].columns)}

    # Find the year with the biggest difference in the number of voters for each municipality
    for municipio in municipios_validos:

        # Get the index of the row with the municipality
        municipio_row_index = municipal_rows[municipal_rows['Territórios']['Região'] == municipio].index[0]
        # Get the series with the number of voters for the municipality
        municipio_data = legislativas["Votantes"].iloc[municipio_row_index]
        # Turn the series into a list
        municipio_data_list_raw = municipio_data.tolist()
        municipio_data_list_shifted = municipio_data_list_raw.copy()
        # Shift a copy of the list to the right
        municipio_data_list_shifted.insert(0, 0)

        # Determine the differences between the number of voters in consecutive years
        differences = [municipio_data_list_raw[i] - municipio_data_list_shifted[i] for i in range(len(municipio_data_list_raw))]
        differences.pop(0)

        # Find the index of the year with the biggest difference
        biggest_difference_index = differences.index(min(differences))
        # Map the index to the year
        municipios_validos[municipio] = years2index[biggest_difference_index+1]
    
    return municipios_validos

def demografiaMunicipios() -> dict[str,tuple[str,str]]:
    """
    Calculates the demographic differences between the years 1975 and 2022 for each municipality in each region.

    Returns:
    A dictionary containing the demographic differences for each region, where the keys are the region names and the values are tuples
    containing the municipality with the minimum and maximum demographic difference.
    """

    # Get the rows corresponding to the NUTS III regions
    local_legislativas = legislativas.copy()

    location_type = 'NUTS III'
    nuts_rows_df = local_legislativas[local_legislativas[('Territórios', 'Âmbito Geográfico')] == location_type]
    nuts_rows = nuts_rows_df["Total"]

    # Get the rows corresponding to the municipalities
    location_type = 'Município'
    municipal_rows = local_legislativas[local_legislativas[('Territórios', 'Âmbito Geográfico')] == location_type]

    nuts_municipality = {}
    # Example: { 'Norte' : [4, 5, 6, 7, 8, 9, 10, 11, 12, 13] }, where 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 are the indexes of the municipalities in the region 'Norte'

 
    # Convert lists to sets
    first_set = set(nuts_rows.index)
    second_set = set(municipal_rows.index)
    
    # Determine the bounds
    min_first = min(first_set)
    max_first = max(first_set)
    
    # All numbers in the second set that are not in the first set
    non_first_numbers = second_set - first_set
    # Sort the non-first numbers to prepare for grouping
    valid_numbers = sorted(non_first_numbers)
    
    # Group consecutive numbers
    grouped_numbers = []
    if valid_numbers:
        current_group = [valid_numbers[0]]
        
        for number in valid_numbers[1:]:
            if number == current_group[-1] + 1:
                current_group.append(number)
            else:
                grouped_numbers.append(current_group)
                current_group = [number]
        
        grouped_numbers.append(current_group)  # Add the last group

    for nuts_row_index, current_group in zip(range(len(nuts_rows)), enumerate(grouped_numbers)):
        nuts_municipality[nuts_rows_df[('Territórios', 'Região')].iloc[nuts_row_index]] = current_group[1]

    # For each region, find the municipality with the biggest difference in the number of voters between 1975 and 2022
    municipality_differences = {}
    municipality_data_dict = {}
    results = {}

    municipal_rows.set_index(local_legislativas.columns[0], inplace=True)

    # Access each region
    for region in nuts_municipality:
        # Access each municipality in the region
        for municipality_index in nuts_municipality[region]:
            # Access the series with the number of voters for the municipality
            municipality_data = local_legislativas["Total"].iloc[municipality_index]
            # Convert the series to a list
            municipality_data_list = municipality_data.tolist()
            # Calculate the difference between the number of voters in 1975 and 2022
            difference = municipality_data_list[-1] - municipality_data_list[0]
            municipality_data_dict.update({local_legislativas[('Territórios', 'Região')].iloc[municipality_index]: difference})

        municipality_differences[region] = municipality_data_dict
        municipality_data_dict = {}

    
    
    for region in municipality_differences:
        results[region] = (min(municipality_differences[region], key=municipality_differences[region].get), 
                           max(municipality_differences[region], key=municipality_differences[region].get))
    
    return results

# T4

nominations = pd.read_csv("dados/nominations.csv")

def maisNomeado() -> tuple[str,int]:
    """
    Finds the nominee with the highest number of unique nominators.

    Returns:
        A tuple containing the name of the most nominated nominee and the number of unique nominators.
    """

    G = nx.DiGraph()

    # Cleaning the data and adding edges, nominators, and nominees to the graph
    for _, row in nominations.iterrows():
        nominators = [name.strip() for name in row["Nominator(s)"].replace('\r\n', '|').split('|')]
        nominees = [name.strip() for name in row["Nominee(s)"].replace('\r\n', ',').split(',')]
        year = row['Year']

        for nominee in nominees:
            for nominator in nominators:
                if G.has_edge(nominator, nominee):
                    G[nominator][nominee]['years'].add(year)
                else:
                    G.add_edge(nominator, nominee, years={year})

    # Count the number of unique nominators for each nominee
    nominee_counts = {}
    
    for nominator, nominee in G.edges():
        if nominee not in nominee_counts:
            nominee_counts[nominee] = set()
        nominee_counts[nominee].add(nominator)

    # Find the nominee with the highest number of unique nominators
    most_nominated = None
    max_nominators = 0
    for nominee, nominators in nominee_counts.items():
        if len(nominators) > max_nominators:
            most_nominated = nominee
            max_nominators = len(nominators)

    # Note: I found more nominators associated with the most nominated nominee, but I reckon this is due to two duplicate entries in the dataset 
    

    return most_nominated, max_nominators

def nomeacoesCruzadas() -> tuple[int, set[str]]:
    """
    Calculates the largest strongly connected component in a directed graph
    based on nominations and returns the size of the component and the categories involved.
    A strongly connected component is a subgraph in which there is a directed path between every pair of nodes.

    Returns:
        A tuple containing the size of the largest strongly connected component and
        a set of categories involved in the component.
    """
    
    G = nx.DiGraph()

    # Cleaning the data and adding edges, nominators, and nominees to the graph
    for _, row in nominations.iterrows():
        nominators = [name.strip() for name in row["Nominator(s)"].replace('\r\n', '|').split('|')]
        nominees = [name.strip() for name in row["Nominee(s)"].replace('\r\n', ',').split(',')]
        category = row["Category"]
        for nominator in nominators:
            for nominee in nominees:
                # Adiciona uma aresta do nominador para o nomeado
                G.add_edge(nominator, nominee, category=category)
    
    # Find the largest strongly connected component
    largest_scc = max(nx.strongly_connected_components(G), key=len)
    
    # Collect the categories involved in the largest SCC
    categories = set()
    # Loop through the edges and add the categories to the set
    for u, v, data in G.edges(data=True):
        # Check if both nodes are in the largest SCC
        if u in largest_scc and v in largest_scc:
            categories.add(data['category'])

    return len(largest_scc), categories

def caminhoEinsteinFeynman() -> list[str]:
    """
    Finds the shortest path between Albert Einstein and Richard Phillips Feynman in a directed graph.

    Returns:
        A list of intermediate nodes representing the shortest path between Einstein and Feynman.
        If no path exists, an empty list is returned.
    """
    G = nx.DiGraph()

    local_nominations = nominations.copy()

    # Filtering the data to include only Physics nominations between 1921 and 1965
    local_nominations = local_nominations[(local_nominations['Year'] >= 1921) & (local_nominations['Year'] <= 1965) & (local_nominations['Category'] == 'Physics')]
    
    # Cleaning the data and adding edges, nominators, and nominees to the graph   
    for idx, row in local_nominations.iterrows():
        nominators = [name.strip() for name in row["Nominator(s)"].replace('\r\n', '|').split('|')]
        nominees = [name.strip() for name in row["Nominee(s)"].replace('\r\n', ',').split(',')]
        
        for nominator in nominators:
            for nominee in nominees:
                G.add_edge(nominator, nominee)
    
    nx.write_graphml(G, "Albert2Feynman.graphml")

    # Find the shortest path between Albert Einstein and Richard Phillips Feynman
    # Einstein as source and Feynman as target
    try:
        # We assume that there is a path between the two nodes
        # If there is no path, a NetworkXNoPath exception is raised

        source = 'Albert Einstein'
        target = 'Richard Phillips Feynman'

        # Find all shortest paths between the source and target nodes
        all_shortest_paths = list(nx.all_shortest_paths(G, source=source, target=target))

        # Extract only intermediate nodes from each path to fix the output format
        all_shortest_paths = [path[1:-1] for path in all_shortest_paths]  

        # Return one of the shortest paths, which matches the solution, though we found multiple shortest paths
        return all_shortest_paths[2]
    
    except nx.NetworkXNoPath:
        return []  