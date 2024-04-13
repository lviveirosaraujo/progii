import json
import pandas as pd
import numpy as np
from numpy import asarray
from PIL import Image # python package Pillow

# T1

with open('dados/prize.json','r') as f:
    prizes = json.load(f)["prizes"]

with open('dados/laureate.json','r') as f:
    laureates = json.load(f)["laureates"]

def maisPartilhados() -> tuple[int,set[tuple[int,str]]]:
    # 1. achar o maximo valor possivel do share
    # 2. indentificar quais são os elementos que possuem o share maximo
    # 3. montar o tuple (ano, categoria) de estas premiaçoes
    # 4. retornar um tuple (share_max,{(ano, categoria),(ano, categoria),(ano, categoria)...})
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
    # remove the economics category
    missing_years = []
    
    # Identify years with missing categories after 1969
    for year in range(min(all_years), max(all_years) + 1):
        if year > 1969:
            if year in categories_by_year:
                if categories_by_year[year] != all_categories:
                    missing_years.append(year)
            else:
                missing_years.append(year)

    all_categories.discard("economics")
    # Identify years with missing categories before 1969
    for year in range(min(all_years), max(all_years) + 1):
        if year <= 1969:
            if year in categories_by_year:
                if categories_by_year[year] != all_categories:
                    missing_years.append(year)
            else:
                missing_years.append(year)

    # Find the longest consecutive range of missing years
    if not missing_years:
        return (0, 0)  # Return a default value if there are no missing years
    
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

    return (longest_start, longest_end)

def rankingDecadas() -> dict[str,tuple[str,int]]:
    from collections import defaultdict

    # Estrutura para armazenar a contagem de laureados por país e década
    decada_pais_count = defaultdict(lambda: defaultdict(int))

    # Percorrendo cada laureado e seus prêmios para contar as afiliações por década
    for laureate in laureates:
        for prize in laureate.get("prizes", []):
            year = int(prize["year"])
            decada = f"{(year // 10)}x"  # Formatando a década como '190x', '191x', etc.
            for affiliation in prize.get("affiliations", []):
                if "country" in affiliation and affiliation["country"]:
                    pais = affiliation["country"]
                    decada_pais_count[decada][pais] += 1  # Incrementar a contagem

    # Estrutura para armazenar o país com mais laureados por década
    resultado = {}
    for decada, paises_count in decada_pais_count.items():
        # Encontrar o país com a maior contagem de laureados na década
        pais_max, count_max = max(paises_count.items(), key=lambda item: item[1])
        resultado[decada] = (pais_max, count_max)

    return resultado

# T2



def toGrayscale(rgb:np.ndarray) -> np.ndarray:
    grayscale = 0.21 * rgb[:, :, 0] + 0.72 * rgb[:, :, 1] + 0.07 * rgb[:, :, 2]
    return grayscale.astype(np.uint8)

def converteGrayscale(fromimg:str,toimg:str) -> None:
    # a 3D numpy array of type uint8
    rgb: np.ndarray = asarray(Image.open(fromimg))
    # a 2D numpy array of type uint8
    grayscale: np.ndarray = toGrayscale(rgb)
    Image.fromarray(grayscale, mode="L").save(toimg)

def toBW(gray:np.ndarray,threshold:tuple[int,int]) -> np.ndarray:
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

def autoThreshold(fromimg:str,tolerance:int) -> tuple[int,int]:
    grayscale: np.ndarray = asarray(Image.open(fromimg))
    grayscale_flattened = grayscale.flatten()
    most_frequent_value = np.argmax(np.bincount(grayscale_flattened))

    lower_bound = max(most_frequent_value - tolerance, 0)
    upper_bound = min(most_frequent_value + tolerance, 255)

    return (lower_bound, upper_bound)

def toContour(bw:np.ndarray) -> np.ndarray:
    contour_img = np.full(bw.shape, 255, dtype=np.uint8)

    bw_right_shifted = np.roll(bw, -1, axis=1)
    is_diff_right = bw != bw_right_shifted
    is_diff_right[:, -1] = False

    bw_down_shifted = np.roll(bw, -1, axis=0)
    is_diff_down = bw != bw_down_shifted
    is_diff_down[-1, :] = False

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

    total = legislativas["Total"].iloc[0]
    eleitores = legislativas["Votantes"].iloc[0]
    porcentagem_de_abstencao.append(((total - eleitores) / total) * 100)

    for index in range(len(legislativas["Total"].columns)):
        abstencao.append((legislativas["Total"].columns[index], porcentagem_de_abstencao[0].iloc[index]))

    return abstencao

def perdaGrandesMunicipios() -> dict[str,int]:

    municipios_validos = {}

    # conseguir o a serie pandas com todos os municipios

    location_type = 'Município'
    municipal_rows = legislativas[legislativas[('Territórios', 'Âmbito Geográfico')] == location_type]

    # primeiro achamos todos os municipios com pelo menos 10000 votantes num dos anos
    for row_index, row_serie in municipal_rows["Votantes"].iterrows():
        for year_index in range(len(municipal_rows["Votantes"].columns)):
            if row_serie.iloc[year_index] >= 10000:
                municipios_validos[legislativas['Territórios']['Região'].iloc[row_index]] = ""
                break
    
    # map the list index to the year
    years2index = {index: year for index, year in enumerate(municipal_rows["Votantes"].columns)}

    # agora vamos achar o ano em que mais perderam votantes em relação às eleições anteriores
    for municipio in municipios_validos:

        # find the row index of the municipio
        municipio_row_index = municipal_rows[municipal_rows['Territórios']['Região'] == municipio].index[0]
        # get the pandas series of the municipio
        municipio_data = legislativas["Votantes"].iloc[municipio_row_index]
        # transform the series into a list
        municipio_data_list_raw = municipio_data.tolist()
        # make a copy of the list
        municipio_data_list_shifted = municipio_data_list_raw.copy()
        # shift the list to the right
        municipio_data_list_shifted.insert(0, 0)

        # differences between the years list
        differences = [municipio_data_list_raw[i] - municipio_data_list_shifted[i] for i in range(len(municipio_data_list_raw))]
        # remove the first element of the list
        differences.pop(0)
        # find the index of the year with the biggest difference
        biggest_difference_index = differences.index(min(differences))
        # map the index to the year
        municipios_validos[municipio] = years2index[biggest_difference_index+1]
    
    return municipios_validos

def demografiaMunicipios() -> dict[str,tuple[str,str]]:
    return None

# T4

nominations = pd.read_csv("dados/nominations.csv")

def maisNomeado() -> tuple[str,int]:
    return None

def nomeacoesCruzadas() -> tuple[int,set[str]]:
    return None

def caminhoEinsteinFeynman() -> list[str]:
    return None