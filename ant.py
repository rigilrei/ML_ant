import math
import random
import numpy as np
import requests
import polyline
import folium
import webbrowser
from typing import List, Tuple, Optional


class RoutePlanner:
    def __init__(self, server_url: str = "http://router.project-osrm.org"):
        self.server_url = server_url

    def get_travel_time(self, start: Tuple[float, float],
                        end: Tuple[float, float],
                        travel_mode: str = 'walking') -> float:
        profile = 'foot' if travel_mode == 'walking' else 'car'
        url = f"{self.server_url}/route/v1/{profile}/{start[1]},{start[0]};{end[1]},{end[0]}"
        params = {'overview': 'false', 'annotations': 'duration'}

        try:
            response = requests.get(url, params=params)
            data = response.json()

            if response.status_code != 200:
                return 60  # Возвращаем значение по умолчанию при ошибке

            duration = data['routes'][0]['duration']
            if travel_mode == 'walking':
                duration *= 1.5  # Добавляем запас для пешего маршрута

            return duration / 60  # Конвертируем секунды в минуты

        except Exception:
            return 60

    def get_route_path(self, points: List[Tuple[float, float]],
                       travel_mode: str = 'walking') -> Optional[List[Tuple[float, float]]]:
        profile = 'foot' if travel_mode == 'walking' else 'car'
        coordinates = ";".join([f"{p[1]},{p[0]}" for p in points])
        url = f"{self.server_url}/route/v1/{profile}/{coordinates}"
        params = {'overview': 'full', 'geometries': 'polyline'}

        try:
            response = requests.get(url, params=params)
            data = response.json()

            if response.status_code != 200:
                return None

            encoded_path = data['routes'][0]['geometry']
            decoded_points = polyline.decode(encoded_path)
            return [(lat, lon) for lon, lat in decoded_points]

        except Exception:
            return None


class RouteOptimizer:
    def __init__(self, locations: List[Tuple[float, float]],
                 location_priorities: List[float],
                 max_travel_time: float,
                 planner: RoutePlanner,
                 ant_count: int = 10,
                 iterations: int = 50,
                 pheromone_decay: float = 0.5,
                 alpha: float = 1,
                 beta: float = 2):

        self.locations = locations
        self.priorities = np.array(location_priorities)
        self.max_time = max_travel_time
        self.planner = planner
        self.ant_count = ant_count
        self.iterations = iterations
        self.decay = pheromone_decay
        self.alpha = alpha
        self.beta = beta

        self.location_count = len(locations)
        self.pheromone = np.ones((self.location_count, self.location_count))
        self.time_matrix = np.zeros((self.location_count, self.location_count))
        self._build_time_matrix()

    def _build_time_matrix(self):
        for i in range(self.location_count):
            for j in range(self.location_count):
                if i != j:
                    self.time_matrix[i][j] = self.planner.get_travel_time(
                        self.locations[i], self.locations[j], 'walking')

    def find_best_route(self) -> Tuple[List[int], float]:
        best_route = []
        best_route_score = -np.inf

        for _ in range(self.iterations):
            routes = self._generate_routes()
            self._update_pheromones(routes)
            current_best_route, current_score = self._get_best_route(routes)

            if current_score > best_route_score:
                best_route = current_best_route
                best_route_score = current_score

        return best_route, best_route_score

    def _generate_routes(self) -> List[List[int]]: # маршруты для всех муравьев
        routes = []
        for _ in range(self.ant_count):
            route = self._build_route()
            if route:
                routes.append(route)
        return routes

    def _build_route(self) -> List[int]: # маршрут для 1 муравья
        route = []
        visited = set()
        total_time = 0

        # Начинаем с точки с максимальным приоритетом
        start_point = np.argmax(self.priorities)
        route.append(start_point)
        visited.add(start_point)

        while len(visited) < self.location_count:
            next_point = self._choose_next_point(route[-1], visited)
            if next_point is None:
                break

            move_time = self.time_matrix[route[-1]][next_point]
            if total_time + move_time > self.max_time:
                break

            route.append(next_point)
            visited.add(next_point)
            total_time += move_time

        return route

    def _choose_next_point(self, current_point: int, visited: set) -> Optional[int]: # вывбираем след точку
        possible_points = [p for p in range(self.location_count) if p not in visited]
        if not possible_points:
            return None

        probabilities = []
        for point in possible_points:
            pheromone = self.pheromone[current_point][point]
            time = max(self.time_matrix[current_point][point], 1)
            heuristic = self.priorities[point] / time
            prob = (pheromone ** self.alpha) * (heuristic ** self.beta)
            probabilities.append(prob)

        total = sum(probabilities)
        if total == 0:
            return random.choice(possible_points)

        probabilities = [p / total for p in probabilities]
        return np.random.choice(possible_points, p=probabilities)

    def _update_pheromones(self, routes: List[List[int]]):
        self.pheromone *= self.decay
        for route in routes:
            route_score = self._calculate_route_score(route)
            for i in range(len(route) - 1):
                from_p, to_p = route[i], route[i + 1]
                self.pheromone[from_p][to_p] += route_score

    def _calculate_route_score(self, route: List[int]) -> float:
        if not route:
            return 0

        total_priority = sum(self.priorities[p] for p in route)
        total_time = sum(self.time_matrix[route[i]][route[i + 1]]
                         for i in range(len(route) - 1))
        time_penalty = max(0, total_time - self.max_time) * 10
        return total_priority - time_penalty

    def _get_best_route(self, routes: List[List[int]]) -> Tuple[List[int], float]:
        best_route = []
        best_score = -np.inf
        for route in routes:
            score = self._calculate_route_score(route)
            if score > best_score:
                best_route = route
                best_score = score
        return best_route, best_score


def show_route_on_map(locations: List[Tuple[float, float]],
                      route_points_indices: List[int],
                      planner: RoutePlanner,
                      travel_mode: str = 'walking'):
    if not route_points_indices:
        return None

    # Создаём карту с центром в первой точке маршрута
    map_center = locations[route_points_indices[0]]
    route_map = folium.Map(location=map_center, zoom_start=13)

    # Добавляем маркеры для всех точек
    for i, location in enumerate(locations):
        color = 'red' if i in route_points_indices else 'gray'
        folium.Marker(
            location=location,
            popup=f"Точка {i} (Приоритет: {location_priorities[i]})",
            icon=folium.Icon(color=color)
        ).add_to(route_map)

    route_locations = [locations[i] for i in route_points_indices]
    folium.PolyLine(
        locations=route_locations,
        color='blue',
        weight=3,
        opacity=0.8
    ).add_to(route_map)
    try:
        exact_route = planner.get_route_path(route_locations, travel_mode)
        if exact_route:
            folium.PolyLine(
                locations=exact_route,
                color='green',
                weight=4,
                opacity=1,
                dash_array='5, 5'
            ).add_to(route_map)
    except Exception:
        pass

    return route_map


if __name__ == "__main__":
    kazan_locations = [
        (55.796127, 49.108795),  # Казанский Кремль
        (55.794229, 49.111369),  # Национальный музей РТ
        (55.790444, 49.134670),  # Парк Победы
        (55.817814, 49.116397),  # Аквапарк Ривьера
        (55.778528, 49.123924),  # ЖК Дворцовая набережная
        (55.798551, 49.106990)  # Центральный стадион
    ]

    location_priorities = [4, 3, 2, 1, 3, 2]

    route_planner = RoutePlanner()
    MAX_TRAVEL_TIME = 180
    ANTS_COUNT = 25
    ITERATIONS = 50

    optimizer = RouteOptimizer(
        locations=kazan_locations,
        location_priorities=location_priorities,
        max_travel_time=MAX_TRAVEL_TIME,
        planner=route_planner,
        ant_count=ANTS_COUNT,
        iterations=ITERATIONS
    )

    optimal_route, route_score = optimizer.find_best_route()

    # Выводим оптимальный маршрут
    print("Оптимальный маршрут:")
    for order, point_idx in enumerate(optimal_route):
        print(f"{order + 1}. Точка {point_idx} (Приоритет: {location_priorities[point_idx]})")

    # Сохраняем и показываем карту
    route_map = show_route_on_map(kazan_locations, optimal_route, route_planner, 'walking')
    if route_map:
        route_map.save("kazan_route.html")
        print("Маршрут сохранён в kazan_route.html")
        webbrowser.open("kazan_route.html")
    else:
        print("Не удалось создать карту маршрута")