import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import random

def get_edges(triangles):
    # Return edges as frozensets (undirected)
    edges = set()
    for tri in triangles:
        for i in range(3):
            a, b = tri[i], tri[(i+1)%3]
            edges.add(frozenset((a, b)))
    return edges

def build_triangle_neighbors(triangles):
    # For each edge, track which triangles share it
    edge_to_tri = {}
    for i, tri in enumerate(triangles):
        for j in range(3):
            a, b = sorted([tri[j], tri[(j+1)%3]])
            edge = (a, b)
            if edge not in edge_to_tri:
                edge_to_tri[edge] = []
            edge_to_tri[edge].append(i)
    return edge_to_tri

def is_convex_quad(p1, p2, p3, p4):
    # Checks whether the quadrilateral p1,p2,p3,p4 is convex
    def cross(a, b, c):
        return (b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])
    return cross(p1, p2, p3) * cross(p3, p4, p1) > 0

def flip_edge(triangles, edge_to_tri, points):
    # Pick a random flippable edge (shared by 2 triangles)
    flippable = [(e, tris) for e, tris in edge_to_tri.items() if len(tris) == 2]
    if not flippable:
        return False
    
    edge, (t1_idx, t2_idx) = random.choice(flippable)
    t1, t2 = triangles[t1_idx], triangles[t2_idx]

    # Get the 4 unique points of the quad
    shared = list(edge)
    other1 = list(set(t1) - set(shared))[0]
    other2 = list(set(t2) - set(shared))[0]

    p1, p2, p3, p4 = points[other1], points[shared[0]], points[other2], points[shared[1]]

    if not is_convex_quad(p1, p2, p3, p4):
        return False  # not convex → can't flip

    # Flip the diagonal
    new_tri1 = [other1, other2, shared[0]]
    new_tri2 = [other1, other2, shared[1]]

    triangles[t1_idx] = new_tri1
    triangles[t2_idx] = new_tri2

    return True

def random_triangulation_with_flips(points, num_flips=1000):
    delaunay = Delaunay(points)
    triangles = delaunay.simplices.tolist()
    for _ in range(num_flips):
        edge_to_tri = build_triangle_neighbors(triangles)
        flip_edge(triangles, edge_to_tri, points)
    return np.array(triangles)

# Usage

figs_dir = 'figs/triangulation/scattered/'

N = 20
np.random.seed(4)
random.seed(4)

x = np.random.rand(N)
y = np.random.rand(N)

# Plot Delaunay
fig, ax = plt.subplots()
ax.triplot(x,y, color='lightcoral', lw=2)
ax.plot(x,y, 'o', color='black')
ax.axis('off')
fig.savefig(figs_dir+'delaunay.svg')

# Plot Flipped
fig, ax = plt.subplots()
points = np.column_stack((x,y))
triangles = random_triangulation_with_flips(points, num_flips=500)
ax.triplot(x,y, triangles, color='lightcoral', lw=2)
ax.plot(x,y, 'o', color='black')
ax.axis('off')
fig.savefig(figs_dir+'arbitrary.svg')

