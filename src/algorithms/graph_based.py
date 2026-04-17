"""Graph-based (GBD) algorithm for binary image rectangle decomposition."""
import networkx as nx
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt

from src.utils.visualization import (
    draw_solution,
    visualize_concave_vertices,
    visualize_concave_vertices_cord,
)


def find_concave_vertices(grid):
    """
    Identify concave vertices in a binary grid using a 2x2 neighborhood analysis.

    A concave vertex (inner corner) is detected when a 2x2 block contains
    exactly three active pixels (value 1). The function returns the top-left
    coordinate of each such 2x2 block.

    Args:
        grid (np.ndarray): A 2D binary numpy array (0s and 1s).

    Returns:
        list[tuple[int, int]]: A list of (x, y) coordinates representing the
            top-left corner of each detected concave vertex block.
    """
    rows, cols = grid.shape
    concave = []

    for y in range(rows - 1):
        for x in range(cols - 1):
            # Extract a 2x2 neighborhood
            block = grid[y:y + 2, x:x + 2]

            # If 3 out of 4 pixels are 1, it's a concave corner
            if np.sum(block) == 3:
                # Store the top-left index of the 2x2 sliding window
                concave.append((x, y))

    return concave


def find_concave_corners(grid):
    """Identify concave corners in a binary grid by analyzing orthogonal and diagonal neighbors.

    A concave corner is identified at an active pixel (1) if two adjacent orthogonal
    neighbors are also active (1), but the diagonal neighbor between them is empty (0).

    Args:
        grid (np.ndarray): 2D binary numpy array (0s and 1s).

    Returns:
        list[tuple]: List of tuples containing (index, x, y, [label], cx, cy),
            where (cx, cy) are the offset coordinates of the concave vertex.
    """
    rows, cols = grid.shape
    concave_points = []
    idx = 0

    for y in range(rows):
        for x in range(cols):
            if grid[y, x] == 1:
                # Helper function to safely read neighbor values
                def g(dy, dx):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < rows and 0 <= nx < cols:
                        return grid[ny, nx]
                    return 0

                # Orthogonal neighbors: top, bottom, left, right
                t, b = g(-1, 0), g(1, 0)
                l, r = g(0, -1), g(0, 1)

                # Diagonal neighbors
                tl, tr = g(-1, -1), g(-1, 1)
                bl, br = g(1, -1), g(1, 1)

                # Check concave corners and assign vertex coordinates (cx, cy)

                # Top-Left concave: Top and Left are filled, but Top-Left diagonal is empty
                if t == 1 and l == 1 and tl == 0:
                    cx, cy = x - 0.5, y - 0.5
                    concave_points.append((idx, x, y, ["top-left"], cx, cy))
                    idx += 1

                # Top-Right concave: Top and Right are filled, but Top-Right diagonal is empty
                if t == 1 and r == 1 and tr == 0:
                    cx, cy = x + 0.5, y - 0.5
                    concave_points.append((idx, x, y, ["top-right"], cx, cy))
                    idx += 1

                # Bottom-Right concave: Bottom and Right are filled, but Bottom-Right diagonal is empty
                if b == 1 and r == 1 and br == 0:
                    cx, cy = x + 0.5, y + 0.5
                    concave_points.append((idx, x, y, ["bottom-right"], cx, cy))
                    idx += 1

                # Bottom-Left concave: Bottom and Left are filled, but Bottom-Left diagonal is empty
                if b == 1 and l == 1 and bl == 0:
                    cx, cy = x - 0.5, y + 0.5
                    concave_points.append((idx, x, y, ["bottom-left"], cx, cy))
                    idx += 1

    return concave_points


def find_cogrid_pairs(concave_vertices, grid):
    """
    Find all co-grid pairs of concave vertices to identify potential internal chords.

    Co-grid pairs are vertices that share the same x or y coordinate. A chord is
    considered valid (an internal cut) only if the path between vertices is
    entirely contained within the object's interior (grid cells on both sides are 1).

    Args:
        concave_vertices (list): List of tuples (id, px, py, corner_type, cx, cy).
        grid (np.ndarray): 2D binary numpy array representing the object.

    Returns:
        list[dict]: List of chord dictionaries containing metadata, endpoints, and range.
    """
    rows, cols = grid.shape
    chords = []
    chord_id = 0

    # 1. Prepare and enrich vertex data for easier access
    enriched_vertices = []
    for vid, px, py, corner, cx, cy in concave_vertices:
        ctype = corner[0] if isinstance(corner, list) else corner
        enriched_vertices.append({
            "id": vid, "px": px, "py": py, "corner": ctype,
            "cx": cx, "cy": cy
        })

    # --- HORIZONTAL CHORDS ---
    # Group vertices by their common Y-coordinate
    y_groups = defaultdict(list)
    for v in enriched_vertices:
        y_groups[v["cy"]].append(v)

    for cy, verts in y_groups.items():
        # Sort by X to ensure visibility check between adjacent vertices
        sorted_verts = sorted(verts, key=lambda v: v["cx"])

        for i in range(len(sorted_verts) - 1):
            v1, v2 = sorted_verts[i], sorted_verts[i + 1]

            # Integer range for grid sampling between vertices
            x_start = int(min(v1["px"], v2["px"]))
            x_end = int(max(v1["px"], v2["px"]))

            # Coordinates for rows immediately above and below the potential chord line
            check_y_above = int(cy - 0.5)
            check_y_below = int(cy + 0.5)

            # Check if the path is entirely inside the object (interior check)
            path_above = False
            if 0 <= check_y_above < rows:
                path_above = all(grid[check_y_above, x] == 1 for x in range(x_start, x_end))

            path_below = False
            if 0 <= check_y_below < rows:
                path_below = all(grid[check_y_below, x] == 1 for x in range(x_start, x_end))

            # Logic: Valid internal chord must have active pixels on BOTH sides.
            # If only one side is 1, it's an external boundary edge, not a cut.
            if path_above and path_below:
                chords.append({
                    "id": chord_id,
                    "type": "horizontal",
                    "v1_id": v1["id"],
                    "v2_id": v2["id"],
                    "v1": (v1["px"], v1["py"]),
                    "v2": (v2["px"], v2["py"]),
                    "y_line": cy,
                    "x_range": (v1["cx"], v2["cx"]),
                    "interior_side": "both"
                })
                chord_id += 1

    # --- VERTICAL CHORDS ---
    # Group vertices by their common X-coordinate
    x_groups = defaultdict(list)
    for v in enriched_vertices:
        x_groups[v["cx"]].append(v)

    for cx, verts in x_groups.items():
        # Sort by Y to ensure visibility check between adjacent vertices
        sorted_verts = sorted(verts, key=lambda v: v["cy"])

        for i in range(len(sorted_verts) - 1):
            v1, v2 = sorted_verts[i], sorted_verts[i + 1]

            # Integer range for grid sampling between vertices
            y_start = int(min(v1["py"], v2["py"]))
            y_end = int(max(v1["py"], v2["py"]))

            # Coordinates for columns immediately left and right of the potential chord line
            check_x_left = int(cx - 0.5)
            check_x_right = int(cx + 0.5)

            path_left = False
            if 0 <= check_x_left < cols:
                path_left = all(grid[y, check_x_left] == 1 for y in range(y_start, y_end))

            path_right = False
            if 0 <= check_x_right < cols:
                path_right = all(grid[y, check_x_right] == 1 for y in range(y_start, y_end))

            # Valid internal chord must have active pixels on BOTH sides
            if path_left and path_right:
                chords.append({
                    "id": chord_id,
                    "type": "vertical",
                    "v1_id": v1["id"],
                    "v2_id": v2["id"],
                    "v1": (v1["px"], v1["py"]),
                    "v2": (v2["px"], v2["py"]),
                    "x_line": cx,
                    "y_range": (v1["cy"], v2["cy"]),
                    "interior_side": "both"
                })
                chord_id += 1

    return chords


def chords_intersect(c1, c2):
    """
    Determine if two chords (one horizontal and one vertical) intersect.

    Chords of the same type (both horizontal or both vertical) are assumed
    not to intersect in this context. For orthogonal chords, the function
    checks if the intersection point of their respective lines falls
    within the range of both chords.

    Args:
        c1 (dict): Dictionary representing the first chord.
        c2 (dict): Dictionary representing the second chord.

    Returns:
        bool: True if the chords intersect or touch, False otherwise.
    """
    # Chords of the same orientation are considered non-intersecting
    if c1['type'] == c2['type']:
        return False

    # Identify which chord is horizontal and which is vertical
    h = c1 if c1['type'] == 'horizontal' else c2
    v = c2 if c1['type'] == 'horizontal' else c1

    # Retrieve positional lines, supporting multiple key naming conventions
    h_y = h.get('y_line') if h.get('y_line') is not None else h.get('y')
    v_x = v.get('x_line') if v.get('x_line') is not None else v.get('x')

    h_x_min, h_x_max = h["x_range"]
    v_y_min, v_y_max = v["y_range"]

    # Check if the vertical line's X coordinate is within the horizontal chord's X range
    intersect_x = (h_x_min <= v_x <= h_x_max)

    # Check if the horizontal line's Y coordinate is within the vertical chord's Y range
    intersect_y = (v_y_min <= h_y <= v_y_max)

    # Intersection occurs if both conditions are met (including endpoints)
    return intersect_x and intersect_y


def build_conflict_graph(chords, grid):
    """
    Construct an adjacency list representing a conflict graph of chords.

    In this graph, each node represents a chord, and an edge between two
    nodes indicates that the corresponding chords intersect (conflict).
    This graph is used to solve the Maximum Independent Set problem for
    optimal polygon decomposition.

    Args:
        chords (list[dict]): A list of chord dictionaries containing geometry and IDs.
        grid (np.ndarray): 2D binary numpy array (unused in current implementation,
            reserved for potential spatial indexing).

    Returns:
        dict: An adjacency list where keys are chord IDs and values are lists of
            intersecting chord IDs.
    """
    # Initialize the graph with an empty list for each chord ID
    graph = {chord['id']: [] for chord in chords}

    # Compare every pair of chords to check for intersections
    for i in range(len(chords)):
        for j in range(i + 1, len(chords)):
            if chords_intersect(chords[i], chords[j]):
                # Add an undirected edge between the two intersecting chords
                graph[chords[i]['id']].append(chords[j]['id'])
                graph[chords[j]['id']].append(chords[i]['id'])

    return graph


def is_bipartite_graph(chords, graph):
    """
    Check if the chord conflict graph is bipartite.

    In the context of orthogonal polygon decomposition, a conflict graph
    formed by horizontal and vertical chords is always bipartite because
    edges only exist between chords of different orientations.

    Args:
        chords (list[dict]): List of chord dictionaries.
        graph (dict): Adjacency list representing chord intersections.

    Returns:
        bool: Always returns True, as horizontal-vertical intersection
            graphs are inherently bipartite.
    """
    return True


def maximum_matching_bipartite(chords, graph):
    """
    Find the maximum matching in the bipartite conflict graph of chords.

    This implementation uses a DFS-based augmenting path algorithm (Kuhn's algorithm)
     to find the maximum number of mutually non-intersecting horizontal and vertical
    chord pairs.

    Args:
        chords (list[dict]): List of chord dictionaries.
        graph (dict): Adjacency list representing chord intersections.

    Returns:
        set[tuple[int, int]]: A set of tuples, each containing a pair of
            matched (intersecting) chord IDs.
    """
    # Filter horizontal chords to serve as the starting set for matching
    horizontal = [c for c in chords if c['type'] == 'horizontal']

    # match_v[v_id] stores the horizontal chord ID currently matched with a vertical chord
    match_v = {}

    def dfs(h_id, visited):
        """
        Attempt to find an augmenting path for the given horizontal chord.
        """
        for v_id in graph[h_id]:
            if v_id in visited:
                continue
            visited.add(v_id)

            # If v_id is not matched or we can find an alternative for its current match
            if v_id not in match_v or dfs(match_v[v_id], visited):
                match_v[v_id] = h_id
                return True
        return False

    # Iterate through all horizontal chords to find maximum matching
    for h_chord in horizontal:
        dfs(h_chord['id'], set())

    # Collect unique matched pairs as sorted tuples of IDs
    matched_pairs = set()
    for v_id, h_id in match_v.items():
        matched_pairs.add(tuple(sorted([h_id, v_id])))

    return matched_pairs


def solve_mis_for_chords(chords, conflict_graph):
    """
    Find a Maximum Independent Set (MIS) of chords using a greedy heuristic.

    Since the conflict graph may contain various types of intersections (H-V,
    and potentially H-H or V-V if they share vertices), this function
    approximates the MIS by prioritizing chords with the lowest degree
    (fewer conflicts) to maximize the number of non-intersecting chords.

    Args:
        chords (list[dict]): List of chord dictionaries.
        conflict_graph (dict): Adjacency list representing conflicts between chords.

    Returns:
        list[int]: A list of chord IDs belonging to the estimated Maximum Independent Set.
    """
    # Sort nodes by their degree (number of conflicts) in ascending order.
    # Nodes with fewer conflicts are prioritized to preserve more chords.
    nodes = sorted(chords, key=lambda c: len(conflict_graph[c['id']]))

    selected_ids = []
    rejected_ids = set()

    for chord in nodes:
        c_id = chord['id']

        # If the chord hasn't been rejected by a previously selected neighbor
        if c_id not in rejected_ids:
            # Include this chord in the independent set
            selected_ids.append(c_id)

            # Mark all immediate neighbors as rejected to avoid conflicts
            for neighbor_id in conflict_graph[c_id]:
                rejected_ids.add(neighbor_id)

    return selected_ids


def is_chord_on_boundary(chord, grid):
    """
    Determine if a chord lies on the boundary of the object.

    A chord is considered a boundary edge if one of its adjacent sides
    (orthogonal to its direction) consists entirely of empty pixels (0).
    This distinguishes internal cutting chords from external perimeter edges.

    Args:
        chord (dict): Dictionary containing chord 'type', 'y' or 'x', and 'x_range' or 'y_range'.
        grid (np.ndarray): 2D binary numpy array (0s and 1s).

    Returns:
        bool: True if the chord is on the boundary, False if it is internal.
    """
    rows, cols = grid.shape

    if chord['type'] == 'horizontal':
        # Use 'y_line' if available, otherwise fallback to 'y'
        y = chord.get('y_line') if chord.get('y_line') is not None else chord.get('y')
        x_min, x_max = chord['x_range']

        # Convert to integer indices for grid sampling
        ix_min, ix_max = int(x_min), int(x_max)
        iy = int(y)

        # Check if the row immediately above or below is entirely empty
        above_empty = all(grid[iy - 1, x] == 0 for x in range(ix_min, ix_max + 1) if iy > 0)
        below_empty = all(grid[iy + 1, x] == 0 for x in range(ix_min, ix_max + 1) if iy < rows - 1)

        return above_empty or below_empty

    else:  # vertical
        # Use 'x_line' if available, otherwise fallback to 'x'
        x = chord.get('x_line') if chord.get('x_line') is not None else chord.get('x')
        y_min, y_max = chord['y_range']

        # Convert to integer indices for grid sampling
        iy_min, iy_max = int(y_min), int(y_max)
        ix = int(x)

        # Check if the column immediately to the left or right is entirely empty
        left_empty = all(grid[y, ix - 1] == 0 for y in range(iy_min, iy_max + 1) if ix > 0)
        right_empty = all(grid[y, ix + 1] == 0 for y in range(iy_min, iy_max + 1) if ix < cols - 1)

        return left_empty or right_empty


def maximum_independent_set(chords, graph, matching_pairs):
    """
    Construct a Maximum Independent Set (MIS) from the bipartite conflict graph.

    This function leverages the result of a maximum bipartite matching to derive
    the MIS. It includes all isolated chords, one chord from each matched pair,
    and any unmatched chords that do not conflict with already selected ones.

    Args:
        chords (list[dict]): List of chord dictionaries.
        graph (dict): Adjacency list representing chord conflicts (intersections).
        matching_pairs (set[tuple]): Set of chord ID pairs representing a maximum matching.

    Returns:
        set[int]: A set of chord IDs forming a Maximum Independent Set.
    """
    # Set of all chord IDs
    all_chord_ids = {c['id'] for c in chords}

    # Extract all chord IDs that are part of the maximum matching
    matched_chord_ids = set()
    for pair in matching_pairs:
        matched_chord_ids.add(pair[0])
        matched_chord_ids.add(pair[1])

    # Identify isolated vertices (chords with no conflicts/edges in the graph)
    isolated = set()
    for chord in chords:
        if len(graph[chord['id']]) == 0:
            isolated.add(chord['id'])

    # Identify chords that were not included in the maximum matching
    unmatched = all_chord_ids - matched_chord_ids

    mis = set()

    # 1. Add isolated vertices: These have no conflicts and always belong in the MIS
    mis.update(isolated)

    # 2. Add one vertex from each matched pair
    # By picking exactly one from each edge in the matching, we resolve those specific conflicts
    for pair in matching_pairs:
        # Deterministically pick the first ID from the sorted tuple
        mis.add(pair[0])

    # 3. Add unmatched vertices if they are independent of the current selection
    for chord_id in unmatched:
        # Check if the chord conflicts with any chord already added to the MIS
        is_independent = True
        for selected_id in mis:
            if selected_id in graph[chord_id]:
                is_independent = False
                break

        if is_independent:
            mis.add(chord_id)

    return mis


def gbd_algorithm_level1(concave_vertices, grid, verbose=False):
    """
    Complete Level 1: Find all possible chords, build a bipartite conflict graph,
    and select the optimal maximum independent set (MIS) of chords.

    This implementation uses Maximum Bipartite Matching (Kőnig's Theorem)
    to guarantee the minimum number of rectangles, matching Gurobi's optimality.
    """
    if verbose:
        print("=" * 70)
        print("GBD ALGORITHM - LEVEL 1 (OPTIMAL MIS)")
        print("=" * 70)

    # 1. Find all possible valid chords between concave vertices
    # This helper should return a list of dicts with 'id', 'type', 'v1_id', 'v2_id', etc.
    chords = find_cogrid_pairs(concave_vertices, grid)

    if not chords:
        if verbose: print("No Level 1 chords found.")
        return [], concave_vertices

    # 2. Separate chords into Horizontal and Vertical sets (the two bipartite partitions)
    h_chords = [c for c in chords if c['type'] == 'horizontal']
    v_chords = [c for c in chords if c['type'] == 'vertical']

    # 3. Build the Bipartite Conflict Graph
    # Nodes are identified by strings like 'h5' or 'v12' to keep partitions distinct
    B = nx.Graph()
    h_nodes = [f"h{c['id']}" for c in h_chords]
    v_nodes = [f"v{c['id']}" for c in v_chords]

    B.add_nodes_from(h_nodes, bipartite=0)
    B.add_nodes_from(v_nodes, bipartite=1)

    for h in h_chords:
        for v in v_chords:
            # We only add an edge if a horizontal chord and vertical chord intersect
            if chords_intersect(h, v):
                B.add_edge(f"h{h['id']}", f"v{v['id']}")

    # 4. Solve for Maximum Matching
    # Providing top_nodes explicitly helps NetworkX handle disconnected components
    matching = nx.bipartite.maximum_matching(B, top_nodes=h_nodes)

    # 5. Find Minimum Vertex Cover (MVC)
    # In bipartite graphs, the Maximum Independent Set (MIS) is the complement of the MVC.
    # Passing the matching explicitly prevents the AmbiguousSolution error.
    mvc = nx.bipartite.to_vertex_cover(B, matching, top_nodes=h_nodes)

    # 6. Select chords that are NOT in the Vertex Cover (this forms the MIS)
    selected_chord_ids = []
    for c in chords:
        node_id = f"{'h' if c['type'] == 'horizontal' else 'v'}{c['id']}"
        if node_id not in mvc:
            selected_chord_ids.append(c['id'])

    level1_chords = [c for c in chords if c['id'] in selected_chord_ids]

    # 7. Identify remaining vertices (those not used as endpoints for Level 1 chords)
    used_v_ids = set()
    for c in level1_chords:
        used_v_ids.add(c['v1_id'])
        used_v_ids.add(c['v2_id'])

    # Filter concave_vertices based on whether their unique ID was used
    # Assuming concave_vertices is a list of tuples/lists where index 0 is the ID
    remaining_vertices = [v for v in concave_vertices if v[0] not in used_v_ids]

    if verbose:
        print(f"Total possible chords: {len(chords)}")
        print(f"MIS selected (Optimal): {len(level1_chords)} chords")
        print(f"Remaining vertices for Level 2: {len(remaining_vertices)}")

    return level1_chords, remaining_vertices


def gbd_algorithm_level1_z(concave_vertices, grid, verbose=False):
    """
    Execute Level 1 of the GBD algorithm to optimize polygon decomposition.

    This stage identifies all valid internal chords between concave vertices,
    constructs a conflict graph based on intersections, and selects a Maximum
    Independent Set (MIS) using a greedy heuristic. This maximizes the number
    of non-intersecting chords to minimize the final number of rectangles.

    Args:
        concave_vertices (list): List of detected concave vertex tuples.
        grid (np.ndarray): 2D binary numpy array representing the object.
        verbose (bool): If True, prints execution details to the console.

    Returns:
        tuple: (level1_chords, remaining_vertices)
            - level1_chords (list[dict]): The selected non-conflicting chords.
            - remaining_vertices (list): Vertices not yet resolved by Level 1 chords.
    """
    if verbose:
        print("=" * 70)
        print("GBD ALGORITHM - LEVEL 1 (GREEDY MIS)")
        print("=" * 70)

    # 1. Find all possible valid horizontal and vertical chords
    chords = find_cogrid_pairs(concave_vertices, grid)

    if not chords:
        if verbose:
            print("No Level 1 chords found.")
        return [], concave_vertices

    # 2. Build the conflict graph
    # Accounts for crossing (H-V) and shared vertex conflicts (H-H, V-V)
    graph = build_conflict_graph(chords, grid)

    # 3. Solve Maximum Independent Set (MIS) using a Greedy Heuristic
    # Priority is given to chords with the lowest degree (fewer conflicts)
    sorted_chords = sorted(chords, key=lambda c: len(graph.get(c['id'], [])))

    selected_chord_ids = []
    forbidden_ids = set()

    for chord in sorted_chords:
        c_id = chord['id']
        if c_id not in forbidden_ids:
            # Add chord to MIS and reject its neighbors
            selected_chord_ids.append(c_id)
            for neighbor_id in graph.get(c_id, []):
                forbidden_ids.add(neighbor_id)

    # 4. Final selection and identification of unresolved vertices
    level1_chords = [c for c in chords if c['id'] in selected_chord_ids]

    used_v_ids = set()
    for c in level1_chords:
        used_v_ids.add(c['v1_id'])
        used_v_ids.add(c['v2_id'])

    # Vertices that were not endpoints of a selected chord are passed to Level 2
    remaining_vertices = [v for v in concave_vertices if v[0] not in used_v_ids]

    if verbose:
        print(f"Total possible chords: {len(chords)}")
        print(f"MIS selected: {len(level1_chords)} chords")
        print(f"Remaining vertices for Level 2: {len(remaining_vertices)}")

    return level1_chords, remaining_vertices


def gbd_algorithm_level2(remaining_vertices, grid, level1_chords, verbose=False):
    """
    Execute Level 2 of the GBD algorithm: Ray-casting to the nearest edge.

    For concave vertices not resolved in Level 1, this function casts rays in
    allowed directions until they hit either the object boundary or an existing
    chord. The shortest possible chord is selected for each vertex.

    Args:
        remaining_vertices (list): Vertices not resolved by Level 1 chords.
        grid (np.ndarray): 2D binary numpy array representing the object.
        level1_chords (list[dict]): Chords already selected in Level 1.
        verbose (bool): If True, prints detailed debug information.

    Returns:
        list[dict]: List of new chords generated in Level 2.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("GBD ALGORITHM - LEVEL 2 (NEAREST EDGE)")
        print("=" * 70)
        print(f"remaining_vertices_count: {len(remaining_vertices)}")

    level2_chords = []

    def get_allowed_directions(corners):
        """Determine valid ray-casting directions based on the concave corner orientation."""
        directions = set()
        c = corners[0] if isinstance(corners, list) else corners
        if "top" in c: directions.add("down")
        if "bottom" in c: directions.add("up")
        if "left" in c: directions.add("right")
        if "right" in c: directions.add("left")
        return directions

    def is_blocked_by_chord(cx, cy, direction, blocking_chords):
        """Check if the current ray position collides with an existing chord."""
        for chord in blocking_chords:
            c_type = chord.get('type')

            if c_type == 'vertical' and direction in ('left', 'right'):
                c_line = chord.get('x_line') if chord.get('x_line') is not None else chord.get('x')
                c_range = chord.get('y_range')

                if c_line is not None and abs(cx - c_line) < 0.01:
                    if c_range and c_range[0] <= cy <= c_range[1]:
                        return True

            elif c_type == 'horizontal' and direction in ('up', 'down'):
                c_line = chord.get('y_line') if chord.get('y_line') is not None else chord.get('y')
                c_range = chord.get('x_range')

                if c_line is not None and abs(cy - c_line) < 0.01:
                    if c_range and c_range[0] <= cx <= c_range[1]:
                        return True
        return False

    def extend_chord(v_cx, v_cy, direction, grid, level2_chords, level1_cuts, verbose=False):
        """
        Extends a ray from a concave vertex to the nearest boundary or existing chord.
        """
        # Combine all existing cuts that can act as a stop barrier for the ray
        blocking_all = level1_cuts + level2_chords
        curr_cx, curr_cy = v_cx, v_cy
        path = [(curr_cx, curr_cy)]
        rows, cols = grid.shape

        # Safety limit to prevent infinite loops
        max_steps = (rows + cols) * 2
        steps = 0

        while steps < max_steps:
            steps += 1

            # 1. MOVEMENT: Step by 0.5 units to check both pixel centers and boundaries
            if direction == 'right':
                curr_cx += 0.5
            elif direction == 'left':
                curr_cx -= 0.5
            elif direction == 'down':
                curr_cy += 0.5
            elif direction == 'up':
                curr_cy -= 0.5

            # 2. HARD BOUNDARY CHECK: Image canvas edges
            if curr_cx < 0 or curr_cx > cols or curr_cy < 0 or curr_cy > rows:
                if verbose:
                    print(f"    value=EXTEND_STOP_REASON:hard_boundary_limit:({curr_cx},{curr_cy})")
                return path

            # 3. PIXEL VALIDATION: Ensure the ray remains inside the object
            # We determine the pixel index based on the direction of travel
            check_x = int(curr_cx - 0.5) if direction == 'left' else int(curr_cx)
            check_y = int(curr_cy - 0.5) if direction == 'up' else int(curr_cy)

            if 0 <= check_x < cols and 0 <= check_y < rows:
                if grid[check_y, check_x] == 0:
                    # Ray hit empty space; stop at the edge of the last valid pixel
                    if verbose:
                        print(f"    value=EXTEND_STOP_REASON:empty_grid_at:({check_x},{check_y})")
                    return path
            else:
                # Ray exited the grid dimensions
                return path

            # 4. CHORD COLLISION: Check if ray intersects an existing Level 1 or Level 2 chord
            # This check is performed only if we are still within a valid '1' pixel
            if is_blocked_by_chord(curr_cx, curr_cy, direction, blocking_all):
                path.append((curr_cx, curr_cy))
                if verbose:
                    print(f"    value=EXTEND_STOP_REASON:chord_collision")
                return path

            # 5. PATH RECORDING: Save coordinates at integer/half-integer intervals
            if curr_cx % 1.0 == 0 or curr_cy % 1.0 == 0:
                if (curr_cx, curr_cy) not in path:
                    path.append((curr_cx, curr_cy))

        return path

    def build_chord(vertex, direction, path):
        """Construct a chord dictionary from the ray path."""
        if len(path) < 2: return None
        idx, px, py, corner, v_cx, v_cy = vertex
        end_cx, end_cy = path[-1]
        ctype = corner[0] if isinstance(corner, list) else corner

        if direction in ('left', 'right'):
            res = {
                'type': 'horizontal', 'v1': (v_cx, v_cy), 'v2': (end_cx, end_cy),
                'y_line': v_cy, 'x_range': (min(v_cx, end_cx), max(v_cx, end_cx)),
                'interior_side': "below" if "top" in ctype else "above",
                'direction': direction, 'v1_id': idx, 'v2_id': None
            }
        else:
            res = {
                'type': 'vertical', 'v1': (v_cx, v_cy), 'v2': (end_cx, end_cy),
                'x_line': v_cx, 'y_range': (min(v_cy, end_cy), max(v_cy, end_cy)),
                'interior_side': "right" if "left" in ctype else "left",
                'direction': direction, 'v1_id': idx, 'v2_id': None
            }
        return res

    # Process each unresolved vertex
    for vertex in remaining_vertices:
        idx, x, y, corners, v_cx, v_cy = vertex
        allowed_directions = get_allowed_directions(corners)

        candidates = []
        for direction in allowed_directions:
            pixels = extend_chord(v_cx, v_cy, direction, grid, level2_chords, level1_chords)
            chord = build_chord(vertex, direction, pixels)
            if chord:
                chord['length'] = len(pixels)
                candidates.append(chord)

        if candidates:
            # Greedy choice: Select the shortest path to minimize decomposition complexity
            selected = min(candidates, key=lambda c: c['length'])
            level2_chords.append(selected)

    return level2_chords


def convert_chords_to_precise_cuts(chords):
    """
    Standardize various chord formats into a precise cut representation.

    This function normalizes chords from both Level 1 and Level 2, ensuring they
    have consistent spatial coordinates ('y_line' or 'x_line') and 'side'
    identifiers. It handles legacy formats by calculating line positions
    based on pixel centers and intended interior directions.

    Args:
        chords (list[dict]): A list of chord dictionaries in either legacy
            or standardized formats.

    Returns:
        list[dict]: A list of normalized dictionaries ready for polygon splitting.
    """
    precise_cuts = []
    for chord in chords:
        if chord['type'] == 'horizontal':
            # Check if y_line already exists (standardized Level 1 or Level 2 format)
            if 'y_line' in chord:
                precise_cuts.append({
                    'type': 'horizontal',
                    'y_line': chord['y_line'],
                    'x_range': chord['x_range'],
                    'side': chord.get('interior_side') or chord.get('direction') or chord.get('side')
                })
            else:
                # Handle legacy format using pixel centers ('y') and offsets
                interior = chord.get('interior_side') or chord.get('direction')
                # Calculate the cut line based on whether the object is above or below the pixel
                y_line = chord['y'] - 0.5 if interior == 'below' else chord['y'] + 0.5
                precise_cuts.append({
                    'type': 'horizontal',
                    'y_line': y_line,
                    'x_range': chord['x_range'],
                    'side': interior
                })
        else:  # vertical
            # Check if x_line already exists
            if 'x_line' in chord:
                precise_cuts.append({
                    'type': 'vertical',
                    'x_line': chord['x_line'],
                    'y_range': chord['y_range'],
                    'side': chord.get('interior_side') or chord.get('direction') or chord.get('side')
                })
            else:
                # Handle legacy format using pixel centers ('x') and offsets
                interior = chord.get('interior_side') or chord.get('direction')
                # Calculate the cut line based on whether the object is to the left or right of the pixel
                x_line = chord['x'] - 0.5 if interior == 'right' else chord['x'] + 0.5
                precise_cuts.append({
                    'type': 'vertical',
                    'x_line': x_line,
                    'y_range': chord['y_range'],
                    'side': interior
                })

    return precise_cuts


def find_rectangles_from_cuts(grid, cuts, verbose=False):
    """
    Decompose a binary grid into rectangles based on specified cuts.

    The function uses flood fill to identify connected regions separated by
    the provided cuts. Each region is then decomposed into a set of
    maximal non-overlapping rectangles using a plane-sweep/histogram approach.

    Args:
        grid (np.ndarray): 2D binary numpy array (0s and 1s).
        cuts (list[dict]): List of normalized cut dictionaries.
        verbose (bool): If True, prints decomposition details and rectangle stats.

    Returns:
        list[dict]: List of dictionaries containing rectangle coordinates and dimensions.
    """
    rows, cols = grid.shape
    region_map = np.full((rows, cols), -1, dtype=int)
    region_id = 0

    def is_cut_between(y1, x1, y2, x2):
        for cut in cuts:
            if cut['type'] == 'vertical':
                x_line = cut['x_line']
                y_start, y_end = cut['y_range']
                if y1 == y2 and abs(x1 - x2) == 1:
                    x_border = (min(x1, x2) + max(x1, x2)) / 2.0
                    result = abs(x_border - x_line) < 0.1 and y_start < y1 < y_end
                    if result:
                        return True
            else:
                y_line = cut['y_line']
                x_start, x_end = cut['x_range']
                if x1 == x2 and abs(y1 - y2) == 1:
                    y_border = (min(y1, y2) + max(y1, y2)) / 2.0
                    result = abs(y_border - y_line) < 0.1 and x_start < x1 < x_end
                    if result:
                        return True
        return False

    def flood_fill(start_y, start_x, rid):
        """Traverse a connected region that is not separated by cuts."""
        queue = deque([(start_y, start_x)])
        region_map[start_y, start_x] = rid
        points = []
        while queue:
            y, x = queue.popleft()
            points.append((x, y))
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if (0 <= ny < rows and 0 <= nx < cols and
                        grid[ny, nx] == 1 and
                        region_map[ny, nx] == -1 and
                        not is_cut_between(y, x, ny, nx)):
                    region_map[ny, nx] = rid
                    queue.append((ny, nx))
        return points

    def largest_rectangle_in_region(points):
        """
        Perform maximal rectangle decomposition within a connected region.

        Uses a greedy approach by repeatedly finding the largest possible
        rectangle within the remaining pixels of the region until all are covered.
        """
        if not points:
            return []

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        W = x_max - x_min + 1
        H = y_max - y_min + 1

        # Create a local mask for the specific region
        local = np.zeros((H, W), dtype=int)
        for (x, y) in points:
            local[y - y_min, x - x_min] = 1

        rectangles = []
        remaining = local.copy()

        def maximal_rectangle_histogram(binary_matrix):
            """
            Standard histogram-stack method to find the largest rectangle in a binary matrix.
            Returns (area, (top_row, left_col, bottom_row, right_col)).
            """
            r_count, c_count = binary_matrix.shape
            height = np.zeros(c_count + 1, dtype=int)
            best_area = 0
            best_rect = None

            for r in range(r_count):
                # Update histogram heights for current row
                for c in range(c_count):
                    height[c] = height[c] + 1 if binary_matrix[r, c] else 0

                # Use stack to find max area in current histogram
                stack = [-1]
                for c in range(c_count + 1):
                    while height[c] < height[stack[-1]]:
                        h = height[stack.pop()]
                        w = c - stack[-1] - 1
                        area = h * w
                        if area > best_area:
                            best_area = area
                            best_rect = (r - h + 1, stack[-1] + 1, r, c - 1)
                    stack.append(c)

            return best_area, best_rect

        # Decompose the region iteratively
        while remaining.sum() > 0:
            area, rect = maximal_rectangle_histogram(remaining)
            if rect is None or area == 0:
                break

            r1, c1, r2, c2 = rect
            ax, ay = c1 + x_min, r1 + y_min
            w_rect, h_rect = c2 - c1 + 1, r2 - r1 + 1

            rectangles.append({
                'min_x': ax,
                'min_y': ay,
                'max_x': ax + w_rect - 1,
                'max_y': ay + h_rect - 1,
                'width': w_rect,
                'height': h_rect,
                'area': w_rect * h_rect,
            })

            # Zero out the selected rectangle and repeat for remaining pixels
            remaining[r1:r2 + 1, c1:c2 + 1] = 0

        return rectangles

    # Iterate through grid to find all isolated regions and decompose them
    all_rectangles = []
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 1 and region_map[i, j] == -1:
                region_points = flood_fill(i, j, region_id)
                region_id += 1
                rects = largest_rectangle_in_region(region_points)
                all_rectangles.extend(rects)

    if verbose:
        print(f"\nTotal Rectangles Found: {len(all_rectangles)}")
        for idx, rect in enumerate(all_rectangles, 1):
            print(f" R{idx}: ({rect['min_x']},{rect['min_y']}) to ({rect['max_x']},{rect['max_y']}) "
                  f"Size: {rect['width']}x{rect['height']}, Area: {rect['area']}px")

    return all_rectangles


def run_graph_based(img, verbose=False):
    """Run graph-based (GBD) decomposition on binary image.

    Main entry point for the graph-based rectangle decomposition algorithm.
    Finds concave corners, applies the complete GBD algorithm, and returns
    a list of non-overlapping rectangles covering all 1-pixels.

    Args:
        img: Binary image to decompose (0s and 1s).
        verbose: Whether to print debug information.

    Returns:
        tuple: (rectangles, generation_history) where:
            - rectangles: List of (x, y, width, height) tuples
            - generation_history: Empty list (for runner compatibility)

    """
    # Find concave corners
    concave_vertices = find_concave_corners(img)

    # Run complete GBD algorithm
    result = gbd_algorithm_complete(concave_vertices, img, verbose)

    # Convert rect dicts to (x, y, width, height) tuples
    rectangles = [
        (int(r['min_x']), int(r['min_y']), int(r['width']), int(r['height']))
        for r in result['rectangles']
    ]

    return rectangles, []


def gbd_algorithm_complete(concave_vertices, grid, verbose=False, debug=False):
    """
    Execute the complete Graph Based Decomposition (GBD) algorithm.

    This function orchestrates the full pipeline:
    1. Analyzes concave vertices.
    2. Applies Level 1 strategy: Finds a Maximum Independent Set of chords to resolve
       concave corners optimally.
    3. Applies Level 2 strategy: Resolves remaining concave vertices by ray-casting
       to the nearest edge or existing chord.
    4. Decomposes the grid into non-overlapping rectangles based on the combined cuts.
    5. Verifies that all original object pixels are covered by the resulting rectangles.

    Args:
        concave_vertices (list): List of tuples containing vertex data (id, x, y, corner_type, cx, cy).
        grid (np.ndarray): 2D binary numpy array (0s and 1s).
        verbose (bool): If True, prints detailed execution logs and summary statistics.

    Returns:
        dict: A results dictionary containing:
            - 'rectangles': List of dictionaries with rectangle geometry.
            - 'level1_chords': Chords selected via the MIS strategy.
            - 'level2_chords': Chords selected via the nearest-edge strategy.
            - 'all_chords': Combined list of all selected chords.
            - 'precise_cuts': Chords converted to standardized cutting line formats.
            - 'coverage': Boolean indicating if the decomposition successfully covers the input grid.
    """
    if verbose:
        print("Concave Vertices: ", len(concave_vertices))
        for idx, x, y, corner, cx, cy in concave_vertices:
            print(f"{idx}: [{x},{y}] corner: {corner} (geom: {cx}, {cy})")

    # 1. Level 1: Find optimal non-intersecting chords using Maximum Independent Set (MIS)
    level1_chords, remaining_vertices = gbd_algorithm_level1(concave_vertices, grid, verbose)

    level1_precise_cuts = convert_chords_to_precise_cuts(level1_chords)
    if debug:
        print("\nLevel 1 Chords: ", len(level1_chords))
        for i in level1_chords:
            print(i)
        print("\nLevel 1 Precise Cuts:")
        for i in level1_precise_cuts:
            print(i)

    # 2. Level 2: Resolve leftover vertices by extending rays to the nearest boundary/chord
    level2_chords = gbd_algorithm_level2(remaining_vertices, grid, level1_chords, verbose)

    if debug:
        print("\nLVL1 Chords: ")
        for i in level1_chords:
            print(i)
        print("\nLVL2 Chords: ")
        for i in level2_chords:
            print(i)

    # 3. Combine results and convert to precise cut lines
    all_chords = level1_chords + level2_chords
    precise_cuts = convert_chords_to_precise_cuts(all_chords)

    if debug:
        print("\nPrecise Cuts: ")
        for i in precise_cuts:
            print(i)

    # 4. Decompose the grid into individual rectangles
    rectangles = find_rectangles_from_cuts(grid, precise_cuts, verbose)

    # 5. Validation: Verify pixel coverage
    original_pixels = set()
    y_coords, x_coords = np.where(grid == 1)
    for x, y in zip(x_coords, y_coords):
        original_pixels.add((x, y))

    covered_pixels = set()
    for rect in rectangles:
        for x in range(int(rect['min_x']), int(rect['max_x']) + 1):
            for y in range(int(rect['min_y']), int(rect['max_y']) + 1):
                covered_pixels.add((x, y))

    is_covered = covered_pixels == original_pixels

    if verbose:
        print("\n" + "=" * 70)
        print("COMPLETE GBD ALGORITHM - SUMMARY")
        print("=" * 70)
        print(f"Total concave vertices: {len(concave_vertices)}")
        print(f"Level 1 (MIS): {len(level1_chords)} cuts")
        print(f"Level 2 (Nearest Edge): {len(level2_chords)} cuts")
        print(f"TOTAL CUTS: {len(all_chords)}")
        print(f"TOTAL RECTANGLES: {len(rectangles)}")
        print(f"Coverage: {len(covered_pixels)}/{len(original_pixels)} "
              f"{'OK' if is_covered else 'ERROR'}")

    return {
        'rectangles': rectangles,
        'level1_chords': level1_chords,
        'level2_chords': level2_chords,
        'all_chords': all_chords,
        'precise_cuts': precise_cuts,
        'coverage': is_covered
    }


def main():
    """Run GBD algorithm demo on sample binary image."""
    # Small test image
    img_small = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    img_bugged_1 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 0, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    img_bugged_2 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    img_bugged_3 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    img_bugged_4 = np.array([
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    img_bugged_5 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    img_holes_1 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 1, 0, 1, 1, 0],
        [0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    img_holes_2 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
        [0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    img_large = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    # Load from dataset (uncomment to use)
    # img_loaded = np.load("../../data/datasets/objects_binary/npy/crown-6_binary.npy")
    # img_loaded = np.load("../../data/datasets/validation/npy/small_75x75_density50.npy")
    # img_loaded = np.load("../../data/datasets/objects_binary/npy/hat-1_binary.npy")
    # img_loaded = np.load("../../data/datasets/research_leafs_binary/npy/Sorbus_aria_1_binary.npy")
    # img_loaded = np.load("../../data/datasets/research_leafs_binary/npy/Vitis_vinifera_3_binary.npy")
    # img_loaded = np.load("../../data/datasets/objects_binary/npy/butterfly-4_binary.npy")
    # img_loaded = np.load("../../data/datasets/objects_binary/npy/crown-3_binary.npy")
    img_loaded = np.load("../../data/datasets/objects_binary/npy/butterfly-14_binary.npy")

    # img_loaded = np.load("../../data/datasets/research_leafs_binary/npy/Ginkgo_biloba_4_binary.npy")

    # img_loaded = np.load("../../data/datasets/research_leafs_binary/npy/Acer_ginnala_2_binary.npy")
    img_loaded = (img_loaded > 0).astype(np.uint8)  # convert 255 → 1
    img = img_loaded

    # Display the image
    plt.title("Binary Image for GBD Decomposition")
    print(np.unique(img, return_counts=True))
    plt.imshow(img, cmap="gray")  # 0=black, 255=white — correct
    plt.axis("off")
    plt.show()

    # Find concave corners and run GBD algorithm
    concave_vertices = find_concave_corners(img)

    # Visualize found concave vertices
    # visualize_concave_vertices(img, concave_vertices)
    # visualize_concave_vertices_cord(img, concave_vertices)

    result = gbd_algorithm_complete(concave_vertices, img, verbose=True)

    # Visualize results
    rect_tuples = [
        (r['min_x'], r['min_y'], r['width'], r['height'])
        for r in result['rectangles']
    ]
    draw_solution(img, rect_tuples, show=True)


if __name__ == "__main__":
    main()