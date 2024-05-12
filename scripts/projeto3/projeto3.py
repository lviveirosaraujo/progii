
import pandas as pd
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
from shapely import *
from shapely.plotting import *
import math
import networkx as nx

# Tarefa 1

# um tuplo (axioma,regras de expansão,ângulo inicial em graus,ângulo de rotação em graus)
lsystem = tuple[str,dict[str,str],float,float]

tree1 : lsystem = ("F",{"F":"F[-F]F[+F][F]"},90,30)
tree2 : lsystem = ("X",{"F":"FF","X":"F-[[X]+X]+F[+FX]-X"},90,22.5)
bush1 : lsystem = ("Y",{"X":"X[-FFF][+FFF]FX","Y":"YFX[+Y][-Y]"},90,25.7)
bush2 : lsystem = ("VZFFF",{"V":"[+++W][---W]YV","W":"+X[-W]Z","X":"-W[+X]Z","Y":"YZ","Z":"[-FFF][+FFF]F"},90,20)
plant1 : lsystem = ("X",{"X":"F+[[X]-X]-F[-FX]+X)","F":"FF"},60,25)

def expandeLSystem(l:lsystem,n:int) -> str:
    axioma, regras, _, _ = l  # Desempacotamos o L-system, ignorando os ângulos
    resultado = axioma  # Inicializamos com o axioma

    for _ in range(n):  # Repetimos a expansão n vezes
        nova_string = ""  # Iniciamos uma nova string para a iteração atual
        for char in resultado:  # Iteramos sobre cada caractere da string atual
            if char in regras:  # Se o caractere tem uma regra de substituição definida
                nova_string += regras[char]  # Substituímos pelo valor correspondente na regra
            else:
                nova_string += char  # Mantemos o caractere se não houver regra de substituição
        resultado = nova_string  # Atualizamos o resultado para a próxima iteração

    return resultado  # Retornamos o resultado após n iterações

def desenhaTurtle(steps:str,start_pos:(float,float),start_angle:float,side:float,theta:float) -> list[list[(float,float)]]:
    pos = start_pos
    angle = start_angle
    lines = [[pos]]
    stack = []
    for s in steps:
        if s=="F":
            pos = (pos[0] + side * math.cos(math.radians(angle)),pos[1] + side * math.sin(math.radians(angle)))
            lines[-1].append(pos)
        elif s=="-": angle = angle-theta
        elif s=="+": angle = angle+theta
        elif s=="[": stack.append((pos,angle))
        elif s=="]": pos,angle = stack.pop() ; lines.append([pos])
    return lines

def desenhaLSystem(l:lsystem,n:int):
    axioma, regras, start_angle, theta = l
    fig, axes = plt.subplots(1, n + 1, figsize=(15, 3))  # Ajusta o tamanho dos subplots

    start_pos = (0, 0)
    side = 10  # Tamanho de cada passo

    # Cada iteração de expansão do L-system
    for i in range(n + 1):
        s = expandeLSystem((axioma, regras, start_angle, theta), i)
        lines = desenhaTurtle(s, start_pos, start_angle, side, theta)
        ax = axes[i]

        depth_stack = []
        depth = 0
        depth_color_map = {}  # Mapeamento de profundidade para linhas

        # Calcula profundidade e guarda linhas com profundidade
        for line in lines:
            for point_index in range(len(line) - 1):
                if point_index == 0 or line[point_index] != line[point_index - 1]:
                    if s[point_index] == '[':
                        depth_stack.append(depth)
                        depth += 1
                    elif s[point_index] == ']':
                        depth = depth_stack.pop()
                
                depth_color_map.setdefault(depth, []).append((line[point_index], line[point_index + 1]))
        
        linewidth = max(0.1, (n - i+1) * 2 / n)  # Grossura da linha decresce com o nível 

        # Desenhar cada segmento de linha com a cor baseada na profundidade
        legend_handles = []  # Para guardar as legendas
        for depth, segments in sorted(depth_color_map.items()):
            color = plt.cm.summer(depth / (max(depth_color_map.keys()) + 1))  # Escolhe uma cor
            # Guarda o primeiro segmento para a legenda
            if depth == max(depth_color_map.keys()):  # Somente adiciona a legenda para a maior profundidade
                line_label = ax.plot([], [], color=color, label=f'Nível {depth}')[0]
                legend_handles.append(line_label)
            for start, end in segments:
                ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=linewidth)

        ax.set_title(f'n={i}')  # Título de cada subplot
        ax.axis('equal')
        ax.set_axis_off()
        if legend_handles:
            ax.legend(handles=legend_handles, loc='upper left')

    plt.tight_layout()
    plt.show()

# Tarefa 2

packaging_waste = pd.read_csv('dados/env_waspac.tsv',na_values=":")
municipal_waste = pd.read_csv('dados/env_wasmun.tsv',na_values=":")

def desenhaReciclagemPaisIndice(ax,pais,indice):

    # desenha um gráfico
    # usar o Axes ax recebido como argumento e não fazer plt.show() aqui
    return None

def testeDesenhaReciclagemPaisIndice():
    _,ax = plt.subplots()
    desenhaReciclagemPaisIndice(ax,'Russia',"packaging")
    plt.show()

def desenhaReciclagem():
    # cria botões e desenha um gráfico chamando desenhaReciclagemPaisIndice
    return None

# Tarefa 3

listings = pd.read_csv('dados/listings.csv')
neighbourhoods = gpd.read_file("dados/neighbourhoods.geojson")

def desenhaZonas():
    """
    Desenha um mapa das zonas com base no número total de reviews por zona.
    
    A função realiza as seguintes etapas:
    1. Calcula o número total de reviews por zona.
    2. Junta os dados com o GeoDataFrame das zonas.
    3. Aplica uma transformação logarítmica nos dados.
    4. Verifica se o DataFrame resultante não está vazio.
    5. Converte o GeoDataFrame para utilizar a projeção correta.
    6. Desenha o mapa com as configurações desejadas.
    
    Returns:
        None
    """
    # preencher
    reviews_por_zona = listings.groupby('neighbourhood')['number_of_reviews'].sum().reset_index()
    
    # Juntar com o GeoDataFrame das zonas
    zonas_com_reviews = neighbourhoods.merge(reviews_por_zona, on='neighbourhood')
    
        # Aplicar a transformação logarítmica
    zonas_com_reviews['log_reviews'] = np.log1p(zonas_com_reviews['number_of_reviews'])
    
    # Verificar se o DataFrame resultante não está vazio
    if zonas_com_reviews.empty:
        print("Não há dados suficientes para desenhar o mapa.")
        return
    
    # Converter o GeoDataFrame para utilizar a projeção correta para adicionar o mapa base
    zonas_com_reviews = zonas_com_reviews.to_crs(epsg=3857)
    
    # Desenhar o mapa
    ax = zonas_com_reviews.plot(column='log_reviews', cmap='cool', alpha=0.8, edgecolor='k', legend=True)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    
    # Configurações adicionais para melhor visualização
    ax.set_axis_off()
    legend = ax.get_legend()
    if legend:
        legend.set_bbox_to_anchor((1, 1))
    ax.set_title('Número Total de Reviews por Zona (log scale)')
    plt.show()

def desenhaAlojamentos():
    # Filtrar alojamentos e bairros no Porto, garantindo que é uma cópia
    porto_listings = listings[listings['neighbourhood_group'] == 'PORTO'].copy()
    porto_neighbourhoods = neighbourhoods[neighbourhoods['neighbourhood_group'] == 'PORTO']
    
    # Definir a figura e o eixo
    fig, ax = plt.subplots(figsize=(25, 15))
    porto_neighbourhoods.plot(ax=ax, color='none', edgecolor='black')
    
    # Aplicar normalização por quantis nos preços e criar mapa de cores
    #porto_listings['price_rank'] = rankdata(porto_listings['price'], method='average') / len(porto_listings['price'])
    #norm = Normalize(vmin=0, vmax=1)
    norm = Normalize(vmin=porto_listings['price'].min(), vmax=porto_listings['price'].max())
    cmap = plt.get_cmap('spring')
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    porto_listings['color'] = porto_listings['price'].apply(lambda x: mappable.to_rgba(x))
    
    # Mapear tipos de alojamento a marcadores
    marker_dict = {'Entire home/apt': '^', 'Private room': 'o', 'Shared room': 's'}

    # Criar handles para a legenda
    handles = [plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor='k', markersize=10, linestyle='None')
               for marker in marker_dict.values()]
    labels = list(marker_dict.keys())
    
    # Plotar usando scatter com parâmetros agrupados por tipo de alojamento
    for room_type, group in porto_listings.groupby('room_type'):
        marker = marker_dict.get(room_type, 'x')  # Define o marcador ou usa 'x' como fallback
        ax.scatter(group['longitude'], group['latitude'],
                   c=group['color'], s=group['availability_365'] / 32,
                   marker=marker, alpha=0.85)
    
    # Adicionar o mapa de fundo e a barra de cores
    ctx.add_basemap(ax, crs=porto_neighbourhoods.crs.to_string(), source=ctx.providers.CartoDB.Positron)
    fig.colorbar(mappable, ax=ax, orientation='horizontal', label='Preço')
    # Adicionar a legenda
    ax.legend(handles, labels, title="Tipo de Alojamento")
    ax.set_title('Alojamentos disponíveis na cidade do Porto')
    
    # Mostrar o gráfico
    plt.show()

def topLocation() -> tuple[str,str,float,float]:
    # Calcular o anfitrião com mais alojamentos
    host_counts = listings['host_id'].value_counts()
    top_host_id = host_counts.idxmax()

    # Filtrar alojamentos do anfitrião mais ativo
    top_host_listings = listings[listings['host_id'] == top_host_id]

    # Converter para GeoDataFrame
    gdf = gpd.GeoDataFrame(top_host_listings, geometry=gpd.points_from_xy(top_host_listings.longitude, top_host_listings.latitude))

    # Ponto central (Porto)
    porto_center = Point(-8.6308, 41.1647)

    # Calcular a distância ao centro do Porto
    gdf['distance_to_center'] = gdf.distance(porto_center)

    # Encontrar o alojamento mais próximo do centro
    min_distance_idx = gdf['distance_to_center'].idxmin()
    closest_listing = gdf.loc[min_distance_idx]

    return closest_listing['name'], closest_listing['host_name'], closest_listing['latitude'], closest_listing['longitude']

def desenhaTop():
    name,host_name,latitude,longitude = topLocation()
    name, host_name, latitude, longitude = topLocation()
    
    # Criar um mapa com a localização do alojamento mais central
    fig, ax = plt.subplots(figsize=(10, 15))

    # Criar GeoDataFrame para o alojamento mais central
    d = {'col1': ['target'], 'geometry': [Point(longitude, latitude)]}
    central_location = gpd.GeoDataFrame(d)
    central_location.set_crs(epsg=4326, inplace=True)

    # Adicionar o mapa base
    central_location.plot(ax=ax, color='red', markersize=20)
    ax.set_aspect('equal')  # Ensuring equal aspect ratio
    # Set axis limits to zoom in around the point of interest
    ax.set_xlim([-8.71, -8.55])
    ax.set_ylim([41.14, 41.21])
    
    print(ax.get_xlim(), ax.get_ylim())

    ctx.add_basemap(ax, crs=central_location.crs, source=ctx.providers.OpenStreetMap.Mapnik)

    # Adicionar rótulo
    ax.text(float(central_location.geometry.x.iloc[0]), float(central_location.geometry.y.iloc[0] + 0.0005), f'{name} ({host_name})', fontsize=12, ha='center')

    plt.show()

# Tarefa 4

bay = pd.read_csv('dados/bay.csv')

def constroiEcosistema() -> nx.DiGraph:
    return None

def desenhaEcosistema():
    g = constroiEcosistema()
    # desenha o grafo
    plt.show()