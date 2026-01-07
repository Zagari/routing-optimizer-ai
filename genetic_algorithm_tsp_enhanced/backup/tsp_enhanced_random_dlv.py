import pygame
from pygame.locals import *
import random
import itertools
import sys
import numpy as np
from typing import List, Dict, Tuple
from genetic_algorithm_enhanced import (
    calculate_fitness_vrp, generate_random_population_vrp,
    vrp_crossover, mutate_vrp, sort_population, tournament_selection,
    calculate_distance
)
from draw_functions_enhanced import draw_vrp_routes, draw_plot, draw_deliveries, draw_depot
from benchmark_att48 import att_48_cities_locations

# Configurações da janela
WIDTH, HEIGHT = 1600, 900
NODE_RADIUS = 8
DEPOT_RADIUS = 15
FPS = 30
PLOT_X_OFFSET = 50
PLOT_Y_OFFSET = 500

# Configurações do Algoritmo Genético
POPULATION_SIZE = 200
MUTATION_PROBABILITY = 0.6
MAX_EPOCHS = 1000
TOURNAMENT_SIZE = 5

# Configurações do Problema VRP
NUM_VEHICLES = 4
VEHICLE_CAPACITY = 200  # kg
VEHICLE_AUTONOMY = 1500  # pixels (distância máxima)

# Cores dos Veículos
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
CYAN = (0, 255, 255)
GRAY = (128, 128, 128)

VEHICLE_COLORS = [BLUE, GREEN, ORANGE, PURPLE, CYAN, RED, YELLOW]

class HospitalDeliverySystem:
    """Sistema de otimização de entregas hospitalares"""
    
    def __init__(self):
        # Configurar localizações baseadas no benchmark att48
        self.setup_locations()
        
        # Criar entregas com prioridades e pesos
        self.create_deliveries()
        
        # Configurar pygame
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Sistema de Otimização de Rotas Hospitalares - VRP com AG")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Inicializar população
        self.population = generate_random_population_vrp(
            self.deliveries, NUM_VEHICLES, POPULATION_SIZE
        )
        
        self.best_fitness_values = []
        self.generation_counter = itertools.count(start=1)
        
    def setup_locations(self):
        """Configura localizações baseadas no benchmark"""
        att_cities = np.array(att_48_cities_locations)
        
        # Escalar para caber na tela
        max_x = max(point[0] for point in att_cities)
        max_y = max(point[1] for point in att_cities)
        
        # Área disponível para o mapa (lado esquerdo da tela)
        map_width = WIDTH - 400  # Reserva espaço para informações
        map_height = HEIGHT - PLOT_Y_OFFSET - 50
        
        scale_x = (map_width - 100) / max_x
        scale_y = (map_height - 100) / max_y
        scale = min(scale_x, scale_y)
        
        self.cities_locations = [
            (int(point[0] * scale + 100), int(point[1] * scale + 50))
            for point in att_cities
        ]
        
        # Depósito central (hospital principal) - primeira localização
        self.depot = self.cities_locations[0]
        
        # Locais de entrega (excluindo o depósito)
        self.delivery_locations = self.cities_locations[1:]
    
    def create_deliveries(self):
        """Cria lista de entregas com prioridades e pesos"""
        self.deliveries = []
        self.priorities = {}
        
        num_deliveries = len(self.delivery_locations)
        
        # Distribuição de prioridades
        num_critical = num_deliveries // 5  # 20% críticos
        num_urgent = num_deliveries // 3    # 33% urgentes
        # Resto é normal
        
        priorities_list = (
            [1] * num_critical +  # Crítico
            [2] * num_urgent +    # Urgente
            [3] * (num_deliveries - num_critical - num_urgent)  # Normal
        )
        random.shuffle(priorities_list)
        
        for i, location in enumerate(self.delivery_locations):
            priority = priorities_list[i]
            
            # Peso baseado na prioridade (medicamentos críticos geralmente são mais leves)
            if priority == 1:  # Crítico
                weight = random.uniform(2, 15)
                item_type = random.choice([
                    "Medicamento Oncológico",
                    "Insulina Especial",
                    "Anticoagulante",
                    "Imunossupressor"
                ])
            elif priority == 2:  # Urgente
                weight = random.uniform(5, 30)
                item_type = random.choice([
                    "Antibióticos",
                    "Analgésicos",
                    "Anti-inflamatórios",
                    "Soro Fisiológico"
                ])
            else:  # Normal
                weight = random.uniform(10, 50)
                item_type = random.choice([
                    "Materiais Cirúrgicos",
                    "EPIs",
                    "Luvas e Máscaras",
                    "Curativos",
                    "Suplementos"
                ])
            
            delivery = {
                'id': i + 1,
                'location': location,
                'priority': priority,
                'weight': weight,
                'item_type': item_type
            }
            
            self.deliveries.append(delivery)
            self.priorities[location] = priority

    def calculate_total_distance(self, routes: List[List[Dict]]) -> float:

        import math
        
        def calc_dist(city1, city2):
            """Calcula distância euclidiana entre duas cidades"""
            return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)
        
        total_distance = 0
        
        for route in routes:
            if not route:
                continue
            
            # Distância do depósito até primeira entrega
            route_distance = calc_dist(self.depot, route[0]['location'])
            
            # Distâncias entre entregas
            for i in range(len(route) - 1):
                route_distance += calc_dist(
                    route[i]['location'], 
                    route[i + 1]['location']
                )
            
            # Distância de volta ao depósito
            route_distance += calc_dist(route[-1]['location'], self.depot)
            
            total_distance += route_distance
        
        return total_distance

    def evolve_generation(self):
        """Executa uma geração do algoritmo genético"""
        # Calcular fitness
        population_fitness = [
            calculate_fitness_vrp(
                individual, self.depot, self.priorities,
                VEHICLE_CAPACITY, VEHICLE_AUTONOMY
            )
            for individual in self.population
        ]
        
        # Ordenar população
        self.population, population_fitness = sort_population(
            self.population, population_fitness
        )
        
        best_fitness = population_fitness[0]
        best_solution = self.population[0]
        
        self.best_fitness_values.append(best_fitness)
        
        # Nova população com elitismo
        elite_size = max(2, POPULATION_SIZE // 10)
        new_population = self.population[:elite_size]
        
        # Gerar novos indivíduos
        while len(new_population) < POPULATION_SIZE:
            # Seleção por torneio
            parent1 = tournament_selection( #Opção alternativa é sort_population
                self.population, population_fitness, TOURNAMENT_SIZE
            )
            parent2 = tournament_selection(
                self.population, population_fitness, TOURNAMENT_SIZE
            )
            
            # Crossover
            child = vrp_crossover(parent1, parent2) #Opção alternativa é order_crossover, pmx_crossover
            
            # Mutação
            child = mutate_vrp(child, MUTATION_PROBABILITY) #Mutação aleatória dos tipos 'swap_within', 'swap_between', 'reverse', 'relocate'
            
            new_population.append(child)
        
        self.population = new_population
        
        return best_fitness, best_solution
    
    def draw_info_panel(self, generation: int, best_fitness: float, 
                    best_solution: List, total_distance: float):
    
        info_x = WIDTH - 380
        info_y = 20
        
        # Fundo do painel
        panel_rect = pygame.Rect(info_x - 10, info_y - 10, 370, HEIGHT - 30)
        pygame.draw.rect(self.screen, (240, 240, 240), panel_rect)
        pygame.draw.rect(self.screen, BLACK, panel_rect, 2)
        
        # Título
        title = self.font.render("SISTEMA DE ROTAS HOSPITALARES", True, BLACK)
        self.screen.blit(title, (info_x, info_y))
        info_y += 35
        
        # Informações da geração
        gen_text = self.small_font.render(f"Geração: {generation}/{MAX_EPOCHS}", True, BLACK)
        self.screen.blit(gen_text, (info_x, info_y))
        info_y += 25
        
        fitness_text = self.small_font.render(f"Melhor Fitness: {best_fitness:.2f}", True, BLUE)
        self.screen.blit(fitness_text, (info_x, info_y))
        info_y += 20
        
        # NOVA LINHA: Distância Total
        distance_text = self.small_font.render(f"Distância Total: {total_distance:.2f} px", True, BLUE)
        self.screen.blit(distance_text, (info_x, info_y))
        info_y += 30
        
        # Estatísticas das rotas
        pygame.draw.line(self.screen, BLACK, (info_x, info_y), (info_x + 350, info_y), 1)
        info_y += 10
        
        stats_title = self.font.render("Estatísticas das Rotas:", True, BLACK)
        self.screen.blit(stats_title, (info_x, info_y))
        info_y += 30
        
        for vehicle_idx, route in enumerate(best_solution):
            if not route:
                continue
            
            color = VEHICLE_COLORS[vehicle_idx % len(VEHICLE_COLORS)]
            
            # Calcular estatísticas da rota
            route_distance = calculate_distance(self.depot, route[0]['location'])
            route_load = 0
            critical_count = 0
            urgent_count = 0
            
            for i, delivery in enumerate(route):
                route_load += delivery['weight']
                if delivery['priority'] == 1:
                    critical_count += 1
                elif delivery['priority'] == 2:
                    urgent_count += 1
                
                if i < len(route) - 1:
                    route_distance += calculate_distance(
                        delivery['location'], route[i + 1]['location']
                    )
            
            route_distance += calculate_distance(route[-1]['location'], self.depot)
            
            # Desenhar informações do veículo
            vehicle_text = self.small_font.render(
                f"Veículo {vehicle_idx + 1}:", True, color
            )
            self.screen.blit(vehicle_text, (info_x, info_y))
            info_y += 20
            
            # Entregas
            deliveries_text = self.small_font.render(
                f"  Entregas: {len(route)}", True, BLACK
            )
            self.screen.blit(deliveries_text, (info_x + 10, info_y))
            info_y += 18
            
            # Prioridades
            priority_text = self.small_font.render(
                f"  Críticas: {critical_count} | Urgentes: {urgent_count}", True, BLACK
            )
            self.screen.blit(priority_text, (info_x + 10, info_y))
            info_y += 18
            
            # Carga
            load_pct = (route_load / VEHICLE_CAPACITY) * 100
            load_color = RED if load_pct > 100 else (GREEN if load_pct > 80 else ORANGE)
            load_text = self.small_font.render(
                f"  Carga: {route_load:.1f}kg ({load_pct:.1f}%)", True, load_color
            )
            self.screen.blit(load_text, (info_x + 10, info_y))
            info_y += 18
            
            # Distância
            dist_pct = (route_distance / VEHICLE_AUTONOMY) * 100
            dist_color = RED if dist_pct > 100 else (GREEN if dist_pct > 80 else ORANGE)
            dist_text = self.small_font.render(
                f"  Distância: {route_distance:.0f}px ({dist_pct:.1f}%)", True, dist_color
            )
            self.screen.blit(dist_text, (info_x + 10, info_y))
            info_y += 25
        
        # Legenda de prioridades
        info_y += 10
        pygame.draw.line(self.screen, BLACK, (info_x, info_y), (info_x + 350, info_y), 1)
        info_y += 10
        
        legend_title = self.font.render("Legenda:", True, BLACK)
        self.screen.blit(legend_title, (info_x, info_y))
        info_y += 25
        
        # Crítico
        pygame.draw.circle(self.screen, RED, (info_x + 10, info_y + 5), 5)
        critical_text = self.small_font.render("Entrega Crítica", True, BLACK)
        self.screen.blit(critical_text, (info_x + 25, info_y))
        info_y += 20
        
        # Urgente
        pygame.draw.circle(self.screen, ORANGE, (info_x + 10, info_y + 5), 5)
        urgent_text = self.small_font.render("Entrega Urgente", True, BLACK)
        self.screen.blit(urgent_text, (info_x + 25, info_y))
        info_y += 20
        
        # Normal
        pygame.draw.circle(self.screen, GREEN, (info_x + 10, info_y + 5), 5)
        normal_text = self.small_font.render("Entrega Normal", True, BLACK)
        self.screen.blit(normal_text, (info_x + 25, info_y))
        info_y += 20
        
        # Depósito
        pygame.draw.circle(self.screen, BLUE, (info_x + 10, info_y + 5), 7)
        depot_text = self.small_font.render("Hospital Central", True, BLACK)
        self.screen.blit(depot_text, (info_x + 25, info_y))

    
    def run(self):
        """Loop principal"""
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
            
            generation = next(self.generation_counter)
            
            if generation > MAX_EPOCHS:
                print(f"\nAlcançou máximo de gerações ({MAX_EPOCHS}). Finalizando.")
                running = False
                continue
            
            self.screen.fill(WHITE)
            
            # Evoluir geração
            best_fitness, best_solution = self.evolve_generation()
            
            # NOVA LINHA: Calcular distância total
            total_distance = self.calculate_total_distance(best_solution)
            
            # Desenhar gráfico de convergência
            draw_plot(
                self.screen,
                list(range(len(self.best_fitness_values))),
                self.best_fitness_values,
                y_label="Fitness",
                position=(PLOT_X_OFFSET, PLOT_Y_OFFSET),
                size=(WIDTH - 450, 350)
            )
            
            # Desenhar rotas
            draw_vrp_routes(
                self.screen, best_solution, self.depot,
                VEHICLE_COLORS, width=2
            )
            
            # Desenhar entregas
            draw_deliveries(
                self.screen, self.deliveries,
                NODE_RADIUS
            )
            
            # Desenhar depósito
            draw_depot(self.screen, self.depot, BLUE, DEPOT_RADIUS)
            
            # Desenhar painel de informações (MODIFICADO: adiciona total_distance)
            self.draw_info_panel(generation, best_fitness, best_solution, total_distance)
            
            # MODIFICADO: Print também mostra distância
            print(f"Geração {generation}: Fitness = {best_fitness:.2f} | Distância = {total_distance:.2f} px")
            
            pygame.display.flip()
            self.clock.tick(FPS)
        
        # Relatório final
        self.generate_final_report(best_solution, best_fitness)
        
        # Manter janela aberta
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        waiting = False
            self.clock.tick(10)
        
        pygame.quit()
        sys.exit()

    
    def generate_final_report(self, best_solution: List, best_fitness: float):

        """Gera relatório final detalhado"""
        print("\n" + "="*80)
        print("RELATÓRIO FINAL - SISTEMA DE OTIMIZAÇÃO DE ROTAS HOSPITALARES")
        print("="*80)
        
        print(f"\nGerações executadas: {min(MAX_EPOCHS, len(self.best_fitness_values))}")
        print(f"Fitness final: {best_fitness:.2f}")
        print(f"Número de veículos utilizados: {len([r for r in best_solution if r])}")
        print(f"Total de entregas: {len(self.deliveries)}")
        
        print("\n" + "-"*80)
        print("DETALHAMENTO DAS ROTAS POR VEÍCULO")
        print("-"*80)
        
        for vehicle_idx, route in enumerate(best_solution):
            if not route:
                continue
            
            print(f"\n### VEÍCULO {vehicle_idx + 1} ###")
            
            # Calcular estatísticas
            route_distance = calculate_distance(self.depot, route[0]['location'])
            route_load = 0
            critical_count = 0
            urgent_count = 0
            normal_count = 0
            
            for i, delivery in enumerate(route):
                route_load += delivery['weight']
                if delivery['priority'] == 1:
                    critical_count += 1
                elif delivery['priority'] == 2:
                    urgent_count += 1
                else:
                    normal_count += 1
                
                if i < len(route) - 1:
                    route_distance += calculate_distance(
                        delivery['location'], route[i + 1]['location']
                    )
            
            route_distance += calculate_distance(route[-1]['location'], self.depot)
            
            print(f"Número de entregas: {len(route)}")
            print(f"  - Críticas: {critical_count}")
            print(f"  - Urgentes: {urgent_count}")
            print(f"  - Normais: {normal_count}")
            print(f"Carga total: {route_load:.2f} kg (Capacidade: {VEHICLE_CAPACITY} kg)")
            print(f"Utilização de carga: {(route_load/VEHICLE_CAPACITY)*100:.1f}%")
            print(f"Distância total: {route_distance:.2f} pixels")
            print(f"Utilização de autonomia: {(route_distance/VEHICLE_AUTONOMY)*100:.1f}%")
            
            # Violações
            violations = []
            if route_load > VEHICLE_CAPACITY:
                violations.append(f"EXCESSO DE CARGA: {route_load - VEHICLE_CAPACITY:.2f} kg")
            if route_distance > VEHICLE_AUTONOMY:
                violations.append(f"EXCESSO DE DISTÂNCIA: {route_distance - VEHICLE_AUTONOMY:.2f} px")
            
            if violations:
                print("\n⚠️  VIOLAÇÕES DE RESTRIÇÕES:")
                for v in violations:
                    print(f"  - {v}")
            else:
                print("\n✓ Todas as restrições respeitadas")
            
            print("\nSequência de entregas:")
            print(f"  0. [DEPÓSITO] Centro de Distribuição")
            for i, delivery in enumerate(route, 1):
                priority_label = {1: "ENTREGA CRÍTICA", 2: "ENTREGA URGENTE", 3: "ENTREGA NORMAL"}[delivery['priority']]
                print(f"  {i}. [ID: {delivery['id']}] {delivery['item_type']} "
                      f"({delivery['weight']:.1f}kg) - Prioridade: {priority_label}")
            print(f"  {len(route)+1}. [RETORNO] Centro de Distribuição")
        
        print("\n" + "="*80)
        print("FIM DO RELATÓRIO")
        print("="*80 + "\n")

if __name__ == "__main__":
    system = HospitalDeliverySystem()
    system.run()