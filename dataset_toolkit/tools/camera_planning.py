import random
import numpy as np
from mathutils import Vector
import math
import sys
from blender_scripts.basic import *

class CameraPlanner:
    """
    RRT in world coordinates:
      - Nodes: world coords (meters)
      - Extend/nearest: world metric
      - Collision: via grid (occupancy) by mapping world->cell
    """
    def __init__(self, terrain_obj, aux_models, start, goal,
                 grid_resolution=0.5, inflate=0.5, eta=0.8, goal_bias=0.5):
        self.terrain_obj = terrain_obj
        self.aux_models = aux_models
        self.grid_resolution = grid_resolution
        self.inflate = inflate
        self.goal_bias = goal_bias
        self.eta = float(eta)  # step (meters) in world coordinates

        # build occupancy grid
        self.grid = self.generate_map()

        # world-space start/goal (x, y)
        self.start_w = (float(start[0]), float(start[1]))
        self.goal_w  = (float(goal[0]),  float(goal[1]))

        # also keep their cells for quick checks if needed
        self.start_c = self.world_to_cell(*self.start_w)
        self.goal_c  = self.world_to_cell(*self.goal_w)

    # ----- coordinate transforms -----
    def world_to_cell(self, x, y):
        c = int((x - self.min_xy[0]) / self.grid_resolution)
        r = int((y - self.min_xy[1]) / self.grid_resolution)
        return max(0, min(self.W - 1, c)), max(0, min(self.H - 1, r))

    def cell_to_world(self, c, r):
        x = self.min_xy[0] + (c + 0.5) * self.grid_resolution
        y = self.min_xy[1] + (r + 0.5) * self.grid_resolution
        return x, y

    # ----- map generation -----
    def generate_map(self):
        bbox_min, bbox_max = object_bbox(self.terrain_obj)
        min_x, min_y, max_x, max_y = bbox_min[0], bbox_min[1], bbox_max[0], bbox_max[1]
        self.min_xy = (float(min_x), float(min_y))
        self.max_xy = (float(max_x), float(max_y))
        self.W = max(1, int((max_x - min_x) / self.grid_resolution))
        self.H = max(1, int((max_y - min_y) / self.grid_resolution))
        grid = np.ones((self.H, self.W), dtype=np.uint8)  # 1=free, 0=blocked

        # mark obstacles (inflated) on grid
        for aux_obj in self.aux_models:
            aux_bbox_min, aux_bbox_max = object_bbox(aux_obj)
            aux_bbox_min = Vector(aux_bbox_min)
            aux_bbox_max = Vector(aux_bbox_max)

            ax0, ay0 = max(aux_bbox_min.x, min_x), max(aux_bbox_min.y, min_y)
            ax1, ay1 = min(aux_bbox_max.x, max_x), min(aux_bbox_max.y, max_y)

            c0, r0 = self.world_to_cell(ax0, ay0)
            c1, r1 = self.world_to_cell(ax1, ay1)

            pad = int(self.inflate / self.grid_resolution)
            c0, r0 = max(0, c0 - pad), max(0, r0 - pad)
            c1, r1 = min(self.W - 1, c1 + pad), min(self.H - 1, r1 + pad)

            # slice is [r0:r1) semantics; ensure proper ordering
            r_lo, r_hi = min(r0, r1), max(r0, r1)
            c_lo, c_hi = min(c0, c1), max(c0, c1)
            grid[r_lo:r_hi+1, c_lo:c_hi+1] = 0

        return grid

    # ----- utils -----
    def _clamp(self, x, y):
        x = min(max(x, self.min_xy[0]), self.max_xy[0])
        y = min(max(y, self.min_xy[1]), self.max_xy[1])
        return x, y

    def is_free(self, x, y):
        c, r = self.world_to_cell(x, y)
        return self.grid[r, c] == 1

    # world-space random sample with goal bias; only returns free samples
    def sampling(self):
        if random.random() < self.goal_bias:
            return self.goal_w
        for _ in range(1000):
            x = random.uniform(self.min_xy[0], self.max_xy[0])
            y = random.uniform(self.min_xy[1], self.max_xy[1])
            if self.is_free(x, y):
                return (x, y)
        # fallback: return start if we somehow fail to find free cell
        return self.start_w

    # collision check between two world points using grid occupancy
    def collision_check(self, p0, p1):
        x0, y0 = p0
        x1, y1 = p1
        dx, dy = (x1 - x0), (y1 - y0)
        dist = max(1e-8, math.hypot(dx, dy))
        # sample about 2 samples per grid cell traversed
        step = max(self.grid_resolution * 0.5, dist / max(2, int(dist / (self.grid_resolution * 0.5))))
        n = max(1, int(dist / step))
        for i in range(n + 1):
            t = i / n
            x = x0 + t * dx
            y = y0 + t * dy
            c, r = self.world_to_cell(x, y)
            if c < 0 or c >= self.W or r < 0 or r >= self.H or self.grid[r, c] == 0:
                return False
        return True

    # ----- RRT in world coordinates -----
    def _nearest_index(self, nodes_world, q):
        qx, qy = q
        # small vectorized speed-up if list is long
        xs = np.array([p[0] for p in nodes_world], dtype=np.float64)
        ys = np.array([p[1] for p in nodes_world], dtype=np.float64)
        idx = int(np.argmin((xs - qx) ** 2 + (ys - qy) ** 2))
        return idx

    def rrt_extend(self, nodes_world, q_rand):
        """
        Extend from nearest node toward q_rand by step eta (meters).
        Returns (new_point_world, parent_index) or (None, None).
        """
        if not nodes_world:
            return None, None
        i_near = self._nearest_index(nodes_world, q_rand)
        x0, y0 = nodes_world[i_near]
        dx, dy = (q_rand[0] - x0), (q_rand[1] - y0)
        dist = math.hypot(dx, dy)
        if dist < 1e-8:
            return None, None

        step = min(self.eta, dist)
        x_new = x0 + (dx / dist) * step
        y_new = y0 + (dy / dist) * step
        x_new, y_new = self._clamp(x_new, y_new)

        p_new = (x_new, y_new)

        # ensure new point is in free space & segment is collision free
        if not self.is_free(*p_new):
            return None, None
        if not self.collision_check((x0, y0), p_new):
            return None, None

        return p_new, i_near

    def find_path(self, parents, nodes_world, end_index):
        path = []
        cur = end_index
        seen = set()
        while cur != -1 and cur not in seen:
            seen.add(cur)
            path.append(nodes_world[cur])
            cur = parents[cur]
        path.reverse()
        return path

    # main API: return a WORLD-SPACE path from start to goal (best-effort)
    def rrt_path_planning(self, max_iter=5000, connect_thresh=2.0):
        # Check start and goal
        print(f"goal_free: {self.is_free(*self.goal_w)}")
        if (not self.is_free(*self.start_w)) or (not self.is_free(*self.goal_w)):
            print()
            print(" -----!!!  rrt_path_planning_mode  cam_planner is not free  !!!------")
            print()
            return [ 1 ]
        # seed with start
        nodes_world = [self.start_w]
        parents = [-1]  # parent index of each node_world
        best_idx = 0
        best_dist = math.hypot(self.start_w[0] - self.goal_w[0], self.start_w[1] - self.goal_w[1])

        for _ in range(max_iter):
            q_rand = self.sampling()
            p_new, i_parent = self.rrt_extend(nodes_world, q_rand)
            if p_new is None:
                continue

            nodes_world.append(p_new)
            parents.append(i_parent)

            # track closest to goal
            d_goal = math.hypot(p_new[0] - self.goal_w[0], p_new[1] - self.goal_w[1])
            if d_goal < best_dist:
                best_dist = d_goal
                best_idx = len(nodes_world) - 1

            # try to directly connect to goal if close enough
            if d_goal < connect_thresh and self.collision_check(p_new, self.goal_w):
                nodes_world.append(self.goal_w)
                parents.append(len(nodes_world) - 2)  # goal's parent is p_new
                return self.find_path(parents, nodes_world, len(nodes_world) - 1)
        return []

        # best-effort fallback: try to connect best node to goal; otherwise return to best
        # if self.collision_check(nodes_world[best_idx], self.goal_w):
        #     nodes_world.append(self.goal_w)
        #     parents.append(best_idx)
        #     return self.find_path(parents, nodes_world, len(nodes_world) - 1)
        # else:
        #     return self.find_path(parents, nodes_world, best_idx)
