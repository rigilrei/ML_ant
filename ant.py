import math
import random
import numpy as np
import requests
import polyline
import folium
from typing import List, Tuple, Optional


class OSRMPlanner:
    def __init__(self, server_url: str = "http://router.project-osrm.org"):
        self.server_url = server_url

    def get_route_time(self, origin: Tuple[float, float],
                       destination: Tuple[float, float],
                       mode: str = 'walking') -> float:
        profile = 'foot' if mode == 'walking' else 'car'
        url = f"{self.server_url}/route/v1/{profile}/{origin[1]},{origin[0]};{destination[1]},{destination[0]}"
        params = {'overview': 'false', 'annotations': 'duration'}

        try:
            response = requests.get(url, params=params)
            data = response.json()

            if response.status_code != 200:
                return 60

            duration = data['routes'][0]['duration']
            if mode == 'walking':
                duration *= 1.5

            return duration / 60

        except Exception:
            return 60

    def get_route_polyline(self, points: List[Tuple[float, float]],
                           mode: str = 'walking') -> Optional[List[Tuple[float, float]]]:
        profile = 'foot' if mode == 'walking' else 'car'
        coordinates = ";".join([f"{p[1]},{p[0]}" for p in points])
        url = f"{self.server_url}/route/v1/{profile}/{coordinates}"
        params = {'overview': 'full', 'geometries': 'polyline'}

        try:
            response = requests.get(url, params=params)
            data = response.json()

            if response.status_code != 200:
                return None

            encoded_polyline = data['routes'][0]['geometry']
            decoded_points = polyline.decode(encoded_polyline)
            return [(lat, lon) for lon, lat in decoded_points]

        except Exception:
            return None


class AntColonyOptimizer:
    def __init__(self, points: List[Tuple[float, float]],
                 priorities: List[float],
                 max_time: float,
                 planner: OSRMPlanner,
                 n_ants: int = 10,
                 n_iterations: int = 50,
                 decay: float = 0.5,
                 alpha: float = 1,
                 beta: float = 2):
        self.points = points
        self.priorities = np.array(priorities)
        self.max_time = max_time
        self.planner = planner
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

        self.n_points = len(points)
        self.pheromone = np.ones((self.n_points, self.n_points))
        self.time_matrix = np.zeros((self.n_points, self.n_points))
        self._compute_time_matrix()

    def _compute_time_matrix(self):
        for i in range(self.n_points):
            for j in range(self.n_points):
                if i != j:
                    self.time_matrix[i][j] = self.planner.get_route_time(
                        self.points[i], self.points[j], 'walking')

    def run(self) -> Tuple[List[int], float]:
        best_path = []
        best_score = -np.inf

        for _ in range(self.n_iterations):
            paths = self._generate_paths()
            self._update_pheromone(paths)
            current_best_path, current_best_score = self._find_best_path(paths)
            if current_best_score > best_score:
                best_path = current_best_path
                best_score = current_best_score

        return best_path, best_score

    def _generate_paths(self) -> List[List[int]]:
        paths = []
        for _ in range(self.n_ants):
            path = self._construct_path()
            if path:
                paths.append(path)
        return paths

    def _construct_path(self) -> List[int]:
        path = []
        visited = set()
        total_time = 0
        start_point = np.argmax(self.priorities)
        path.append(start_point)
        visited.add(start_point)

        while len(visited) < self.n_points:
            next_point = self._select_next_point(path[-1], visited)
            if next_point is None:
                break

            move_time = self.time_matrix[path[-1]][next_point]
            if total_time + move_time > self.max_time:
                break

            path.append(next_point)
            visited.add(next_point)
            total_time += move_time

        return path

    def _select_next_point(self, current_point: int,
                           visited: set) -> Optional[int]:
        available = [p for p in range(self.n_points) if p not in visited]
        if not available:
            return None

        probabilities = []
        for point in available:
            pheromone = self.pheromone[current_point][point]
            time = max(self.time_matrix[current_point][point], 1)
            heuristic = self.priorities[point] / time
            prob = (pheromone ** self.alpha) * (heuristic ** self.beta)
            probabilities.append(prob)

        total = sum(probabilities)
        if total == 0:
            return random.choice(available)

        probabilities = [p / total for p in probabilities]
        return np.random.choice(available, p=probabilities)

    def _update_pheromone(self, paths: List[List[int]]):
        self.pheromone *= self.decay
        for path in paths:
            path_score = self._calculate_path_score(path)
            for i in range(len(path) - 1):
                from_p, to_p = path[i], path[i + 1]
                self.pheromone[from_p][to_p] += path_score

    def _calculate_path_score(self, path: List[int]) -> float:
        if not path:
            return 0

        total_priority = sum(self.priorities[p] for p in path)
        total_time = sum(self.time_matrix[path[i]][path[i + 1]]
                         for i in range(len(path) - 1))
        time_penalty = max(0, total_time - self.max_time) * 10
        return total_priority - time_penalty

    def _find_best_path(self, paths: List[List[int]]) -> Tuple[List[int], float]:
        best_path = []
        best_score = -np.inf
        for path in paths:
            score = self._calculate_path_score(path)
            if score > best_score:
                best_path = path
                best_score = score
        return best_path, best_score


def visualize_route(points: List[Tuple[float, float]],
                    route_indices: List[int],
                    planner: OSRMPlanner,
                    mode: str = 'walking'):
    if not route_indices:
        return None

    m = folium.Map(location=points[route_indices[0]], zoom_start=13)

    for i, point in enumerate(points):
        color = 'red' if i in route_indices else 'gray'
        folium.Marker(
            location=point,
            popup=f"Point {i} (Priority: {priorities[i]})",
            icon=folium.Icon(color=color)
        ).add_to(m)

    route_points = [points[i] for i in route_indices]

    folium.PolyLine(
        locations=route_points,
        color='blue',
        weight=3,
        opacity=0.8
    ).add_to(m)

    try:
        route_line = planner.get_route_polyline(route_points, mode)
        if route_line:
            folium.PolyLine(
                locations=route_line,
                color='green',
                weight=4,
                opacity=1,
                dash_array='5, 5'
            ).add_to(m)
    except Exception:
        pass

    return m


if __name__ == "__main__":
    points = [
        (55.796127, 49.108795),  # Казанский Кремль
        (55.782454, 49.122141),  # Улица Баумана
        (55.790939, 49.114654),  # Мечеть Кул Шариф
        (55.800305, 49.106395),  # Центральный парк им. Горького
    ]
    priorities = [1, 4, 3, 1]

    planner = OSRMPlanner()
    MAX_TIME = 120
    N_ANTS = 20
    N_ITERATIONS = 30

    optimizer = AntColonyOptimizer(
        points=points,
        priorities=priorities,
        max_time=MAX_TIME,
        planner=planner,
        n_ants=N_ANTS,
        n_iterations=N_ITERATIONS
    )

    best_path, best_score = optimizer.run()

    for i, point_idx in enumerate(best_path):
        print(f"{i + 1}. Point {point_idx} (Priority: {priorities[point_idx]})")

    route_map = visualize_route(points, best_path, planner, 'walking')
    if route_map:
        route_map.save("optimal_route.html")
        print("Route saved to optimal_route.html")
    else:
        print("Failed to visualize route")
