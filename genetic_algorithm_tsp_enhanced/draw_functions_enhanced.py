import pygame
from typing import List, Tuple, Dict
import math

def draw_paths(screen, route: List[Tuple[int, int]], rgb_color: Tuple[int, int, int], width: int = 2):
    """Desenha caminho para TSP simples"""
    if len(route) < 2:
        return
    
    for i in range(len(route) - 1):
        pygame.draw.line(screen, rgb_color, route[i], route[i + 1], width)
    
    # Fecha o ciclo
    pygame.draw.line(screen, rgb_color, route[-1], route[0], width)

def draw_vrp_routes(screen, routes: List[List[Dict]], depot: Tuple[int, int], 
                    colors: List[Tuple[int, int, int]], width: int = 2):
    """Desenha rotas para VRP com múltiplos veículos"""
    for vehicle_idx, route in enumerate(routes):
        if not route:
            continue
        
        color = colors[vehicle_idx % len(colors)]
        
        # Linha do depósito para primeira entrega
        pygame.draw.line(screen, color, depot, route[0]['location'], width)
        
        # Linhas entre entregas
        for i in range(len(route) - 1):
            pygame.draw.line(
                screen, color,
                route[i]['location'],
                route[i + 1]['location'],
                width
            )
        
        # Linha da última entrega de volta ao depósito
        pygame.draw.line(screen, color, route[-1]['location'], depot, width)

def draw_cities(screen, cities: List[Tuple[int, int]], color: Tuple[int, int, int], radius: int):
    """Desenha cidades para TSP"""
    for city in cities:
        pygame.draw.circle(screen, color, city, radius)
        pygame.draw.circle(screen, (0, 0, 0), city, radius, 1)

def draw_deliveries(screen, deliveries: List[Dict], radius: int):
    """Desenha pontos de entrega com cores baseadas na prioridade"""
    priority_colors = {
        1: (255, 0, 0),      # Crítico - Vermelho
        2: (255, 165, 0),    # Urgente - Laranja
        3: (0, 255, 0)       # Normal - Verde
    }
    
    for delivery in deliveries:
        color = priority_colors.get(delivery['priority'], (128, 128, 128))
        location = delivery['location']
        
        # Círculo preenchido
        pygame.draw.circle(screen, color, location, radius)
        # Borda preta
        pygame.draw.circle(screen, (0, 0, 0), location, radius, 2)

def draw_depot(screen, depot: Tuple[int, int], color: Tuple[int, int, int], radius: int):
    """Desenha o depósito central (hospital)"""
    # Círculo maior para o depósito
    pygame.draw.circle(screen, color, depot, radius)
    pygame.draw.circle(screen, (0, 0, 0), depot, radius, 3)
    
    # Desenha uma cruz (símbolo de hospital)
    cross_size = radius // 2
    pygame.draw.line(
        screen, (255, 255, 255),
        (depot[0] - cross_size, depot[1]),
        (depot[0] + cross_size, depot[1]),
        3
    )
    pygame.draw.line(
        screen, (255, 255, 255),
        (depot[0], depot[1] - cross_size),
        (depot[0], depot[1] + cross_size),
        3
    )

def draw_plot(screen, x_data: List[float], y_data: List[float], 
              y_label: str = "Fitness", 
              position: Tuple[int, int] = (50, 500),
              size: Tuple[int, int] = (700, 350)):
    """Desenha gráfico de convergência"""
    if not y_data:
        return
    
    plot_x, plot_y = position
    plot_width, plot_height = size
    
    # Fundo do gráfico
    plot_rect = pygame.Rect(plot_x, plot_y, plot_width, plot_height)
    pygame.draw.rect(screen, (240, 240, 240), plot_rect)
    pygame.draw.rect(screen, (0, 0, 0), plot_rect, 2)
    
    # Margens internas
    margin = 40
    graph_x = plot_x + margin
    graph_y = plot_y + margin
    graph_width = plot_width - 2 * margin
    graph_height = plot_height - 2 * margin
    
    # Encontrar min e max para escala
    min_y = min(y_data)
    max_y = max(y_data)
    range_y = max_y - min_y if max_y != min_y else 1
    
    # Desenhar eixos
    pygame.draw.line(
        screen, (0, 0, 0),
        (graph_x, graph_y + graph_height),
        (graph_x + graph_width, graph_y + graph_height),
        2
    )
    pygame.draw.line(
        screen, (0, 0, 0),
        (graph_x, graph_y),
        (graph_x, graph_y + graph_height),
        2
    )
    
    # Labels
    font = pygame.font.Font(None, 20)
    
    # Label Y
    y_label_text = font.render(y_label, True, (0, 0, 0))
    screen.blit(y_label_text, (plot_x + 5, plot_y + 5))
    
    # Valores min e max no eixo Y
    max_text = font.render(f"{max_y:.0f}", True, (0, 0, 0))
    min_text = font.render(f"{min_y:.0f}", True, (0, 0, 0))
    screen.blit(max_text, (graph_x - 35, graph_y - 5))
    screen.blit(min_text, (graph_x - 35, graph_y + graph_height - 10))
    
    # Label X
    x_label_text = font.render("Geração", True, (0, 0, 0))
    screen.blit(x_label_text, (graph_x + graph_width - 60, graph_y + graph_height + 15))
    
    # Desenhar linha do gráfico
    if len(y_data) > 1:
        points = []
        for i, y in enumerate(y_data):
            x = graph_x + (i / (len(y_data) - 1)) * graph_width
            y_scaled = graph_y + graph_height - ((y - min_y) / range_y) * graph_height
            points.append((x, y_scaled))
        
        if len(points) >= 2:
            pygame.draw.lines(screen, (0, 0, 255), False, points, 2)
            
            # Desenhar pontos
            for point in points[::max(1, len(points)//50)]:  # Desenha alguns pontos
                pygame.draw.circle(screen, (255, 0, 0), (int(point[0]), int(point[1])), 3)
