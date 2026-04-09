# E01: Graph Theory Basics — Copilot Directive

## Metadata

- **Notebook ID**: E01
- **Title**: Graph Theory Basics
- **Branch**: `feature/E01_graph_theory_basics`
- **Duration**: 65 minutes
- **Prerequisites**: C02 (Connectivity Matrices)
- **Next**: E02 (Introduction to Hyperscanning)

---

## Learning Objectives

By the end of this notebook, learners will be able to:
1. Represent connectivity matrices as graphs (nodes and edges)
2. Understand weighted vs binary graphs
3. Compute basic node metrics (degree, strength, clustering coefficient)
4. Compute global graph metrics (density, path length, efficiency)
5. Identify network hubs and their importance
6. Visualize brain networks effectively
7. Apply graph concepts to hyperscanning (dual-brain networks)

---

## Required Imports
```
numpy, matplotlib, scipy.sparse, networkx (optional, for comparison)
typing, numpy.typing
Local: src/colors.py, src/plotting.py, src/connectivity.py
New: src/graph.py
```

---

## Section 1: Introduction — From Matrices to Networks

**Purpose**: Motivate graph-theoretic analysis of connectivity.

**Content to cover**:
- Connectivity matrices are rich with information
- But how to SUMMARIZE them?
- **Graph theory**: mathematical framework for analyzing networks
- A connectivity matrix IS a graph in disguise!
- Brain networks: channels = nodes, connectivity = edges
- Graph metrics provide interpretable summaries
- Applications:
  - Identify "hub" regions
  - Characterize network organization
  - Compare conditions (rest vs task)
  - Track development/disease
- For hyperscanning: analyze dual-brain networks as unified graph

**Key message**: "Graph theory transforms connectivity matrices into interpretable network summaries."

**Length**: 4-5 paragraphs

---

## Section 2: Graphs — Nodes and Edges

**Purpose**: Core definitions.

**Content to cover**:
- **Graph G = (V, E)**:
  - V = set of vertices (nodes)
  - E = set of edges (connections)
- **For EEG**:
  - Nodes = electrodes/channels
  - Edges = significant connectivity between pairs
- **Adjacency matrix A**:
  - A[i, j] = 1 if edge exists between i and j
  - A[i, j] = 0 otherwise
  - Our connectivity matrix → adjacency matrix!
- **Types of graphs**:
  - **Undirected**: edges have no direction (A symmetric)
  - **Directed**: edges have direction (A may be asymmetric)
  - **Binary**: edges present or absent (0/1)
  - **Weighted**: edges have strengths (continuous values)
- Most connectivity metrics (PLV, coherence) → weighted undirected
- Transfer entropy → weighted directed

**Visualization 1**:
- Simple graph example:
  - 6 nodes arranged in circle
  - Several edges connecting them
  - Corresponding adjacency matrix beside it
- Caption: "A graph and its adjacency matrix representation"

**Visualization 2**:
- Four graph types in 2×2 grid:
  - Binary undirected
  - Binary directed
  - Weighted undirected
  - Weighted directed
- Each with small adjacency matrix
- Caption: "Types of graphs"

---

## Section 3: From Connectivity Matrix to Graph

**Purpose**: Practical conversion.

**Content to cover**:
- Connectivity matrices ARE weighted adjacency matrices
- **Thresholding**: convert weighted → binary
  - Keep only edges above threshold
  - Choices: absolute value, proportional (keep top X%), significance-based
- **Why threshold?**
  - Some metrics require binary graphs
  - Remove weak/spurious connections
  - Focus on strongest connections
- **Trade-offs**:
  - High threshold → sparse graph, may lose structure
  - Low threshold → dense graph, may include noise
- **Proportional thresholding**: keep same density across subjects
  - Important for group comparisons
  - Ensures equal edge count

**Code to produce**:

Function 1: `threshold_matrix_absolute`
```
Parameters:
    matrix: NDArray[np.float64]
    threshold: float
Returns:
    NDArray[np.float64]
Binary adjacency matrix: 1 where |matrix| > threshold, 0 otherwise
Diagonal set to 0
```

Function 2: `threshold_matrix_proportional`
```
Parameters:
    matrix: NDArray[np.float64]
    density: float  # proportion of edges to keep (0-1)
Returns:
    NDArray[np.float64]
Binary adjacency matrix keeping top 'density' proportion of edges
```

Function 3: `get_graph_density`
```
Parameters:
    adjacency: NDArray[np.float64]
Returns:
    float
Density = actual edges / possible edges
For n nodes: density = sum(A) / (n * (n-1))
```

**Visualization 3**:
- Weighted connectivity matrix
- Arrow showing thresholding
- Resulting binary adjacency matrix
- Graph visualization of binary network
- Caption: "Converting connectivity matrix to binary graph"

**Visualization 4**:
- Same matrix with different thresholds:
  - Low threshold (dense graph)
  - Medium threshold
  - High threshold (sparse graph)
- Show resulting graphs side by side
- Caption: "Effect of threshold on network density"

---

## Section 4: Node Metrics — Degree and Strength

**Purpose**: Characterize individual nodes.

**Content to cover**:
- **Degree** (binary graphs):
  - Number of edges connected to node
  - k_i = Σ_j A[i, j]
  - High degree = many connections = potential hub
- **Strength** (weighted graphs):
  - Sum of edge weights connected to node
  - s_i = Σ_j W[i, j]
  - High strength = strong total connectivity
- **In vs Out degree** (directed graphs):
  - In-degree: edges coming IN to node
  - Out-degree: edges going OUT from node
  - For TE: high out-degree = information broadcaster
- **Degree distribution**:
  - Histogram of degrees across all nodes
  - Reveals network topology
  - Random vs scale-free vs regular

**Code to produce**:

Function 4: `compute_degree`
```
Parameters:
    adjacency: NDArray[np.float64]  # binary
Returns:
    NDArray[np.float64]
Degree of each node (for undirected: row sum)
```

Function 5: `compute_strength`
```
Parameters:
    weighted_matrix: NDArray[np.float64]
Returns:
    NDArray[np.float64]
Strength of each node (sum of edge weights)
```

Function 6: `compute_in_out_degree`
```
Parameters:
    adjacency: NDArray[np.float64]  # directed
Returns:
    Tuple[NDArray[np.float64], NDArray[np.float64]]
    (in_degree, out_degree)
For directed graphs
```

**Visualization 5**:
- Example network with nodes sized by degree
- Color intensity by strength
- Highlight high-degree nodes (hubs)
- Caption: "Node degree and strength visualization"

**Visualization 6**:
- Degree distribution histogram
- Compare different network types:
  - Random network (Poisson-like)
  - Scale-free network (power-law)
  - Regular lattice (uniform)
- Caption: "Degree distributions reveal network topology"

---

## Section 5: Clustering Coefficient — Local Structure

**Purpose**: Quantify local connectivity patterns.

**Content to cover**:
- **Question**: Are a node's neighbors also connected to each other?
- **Clustering coefficient C_i**:
  - Proportion of possible triangles that actually exist
  - C_i = (triangles around i) / (possible triangles around i)
  - Range: 0 (no clustering) to 1 (fully clustered)
- **Intuition**: 
  - High C: "cliquey" neighborhood, friends know each other
  - Low C: neighbors don't connect, star-like structure
- **For brain networks**:
  - High clustering suggests local processing
  - Functional modules have high internal clustering
- **Global clustering**: average across all nodes
  - C = (1/n) Σ_i C_i

**Code to produce**:

Function 7: `compute_clustering_coefficient`
```
Parameters:
    adjacency: NDArray[np.float64]  # binary
Returns:
    NDArray[np.float64]
Local clustering coefficient for each node
Handle nodes with degree < 2 (return 0 or NaN)
```

Function 8: `compute_clustering_weighted`
```
Parameters:
    weighted_matrix: NDArray[np.float64]
Returns:
    NDArray[np.float64]
Weighted clustering coefficient (Onnela et al. formula)
Uses geometric mean of triangle weights
```

Function 9: `compute_global_clustering`
```
Parameters:
    adjacency: NDArray[np.float64]
Returns:
    float
Average clustering coefficient across nodes
```

**Visualization 7**:
- Two example networks:
  - High clustering (many triangles)
  - Low clustering (star-like, few triangles)
- Triangles highlighted in color
- Caption: "High vs low clustering networks"

**Visualization 8**:
- EEG-like network with nodes colored by clustering coefficient
- Topographic layout (electrode positions)
- Identify regions with high local clustering
- Caption: "Clustering coefficient across brain regions"

---

## Section 6: Path Length and Efficiency

**Purpose**: Global integration metrics.

**Content to cover**:
- **Shortest path**: minimum number of edges between two nodes
- **Characteristic path length L**:
  - Average shortest path across all node pairs
  - L = (1 / n(n-1)) Σ_{i≠j} d(i, j)
  - Low L = information travels quickly = integrated network
- **Problem**: disconnected nodes have infinite path length!
- **Solution**: Global efficiency E
  - E = (1 / n(n-1)) Σ_{i≠j} 1/d(i, j)
  - Harmonic mean of inverse distances
  - Handles disconnected nodes naturally (1/∞ = 0)
  - High E = efficient network
- **For brain networks**:
  - High efficiency suggests good integration
  - Compare rest vs task
  - Track changes with development/disease

**Code to produce**:

Function 10: `compute_shortest_paths`
```
Parameters:
    adjacency: NDArray[np.float64]
Returns:
    NDArray[np.float64]
Matrix of shortest path lengths between all pairs
Use Floyd-Warshall or BFS
Disconnected pairs: np.inf
```

Function 11: `compute_characteristic_path_length`
```
Parameters:
    adjacency: NDArray[np.float64]
Returns:
    float
Average shortest path (exclude disconnected pairs)
Return np.inf if graph is disconnected
```

Function 12: `compute_global_efficiency`
```
Parameters:
    adjacency: NDArray[np.float64]
Returns:
    float
Global efficiency (harmonic mean of inverse paths)
Handles disconnected nodes properly
```

Function 13: `compute_local_efficiency`
```
Parameters:
    adjacency: NDArray[np.float64]
Returns:
    NDArray[np.float64]
Local efficiency for each node (efficiency of subgraph of neighbors)
```

**Visualization 9**:
- Example network with path highlighted
- Show shortest path between two distant nodes
- Annotate path length
- Caption: "Shortest path in a network"

**Visualization 10**:
- Two networks:
  - High efficiency (many shortcuts)
  - Low efficiency (long paths)
- Color nodes by local efficiency
- Caption: "Network efficiency comparison"

---

## Section 7: Small-World Networks

**Purpose**: The "sweet spot" network organization.

**Content to cover**:
- **Observation**: Brain networks are neither random nor regular
- **Small-world networks** (Watts & Strogatz, 1998):
  - High clustering (like regular lattice)
  - Short path length (like random network)
  - Best of both worlds!
- **Small-worldness σ**:
  - σ = (C / C_random) / (L / L_random)
  - σ > 1 indicates small-world topology
  - Typical brain networks: σ ≈ 2-5
- **Why small-world?**
  - Local clustering: specialized processing
  - Short paths: efficient integration
  - Optimal for information processing
- **Generate reference networks**:
  - Random: same nodes/edges, randomly rewired
  - Lattice: regular local connections only

**Code to produce**:

Function 14: `generate_random_graph`
```
Parameters:
    n_nodes: int
    density: float
    seed: int | None = None
Returns:
    NDArray[np.float64]
Random (Erdős-Rényi) graph with given density
```

Function 15: `generate_lattice_graph`
```
Parameters:
    n_nodes: int
    k_neighbors: int  # connect to k nearest neighbors
Returns:
    NDArray[np.float64]
Regular lattice (ring) graph
```

Function 16: `compute_small_worldness`
```
Parameters:
    adjacency: NDArray[np.float64]
    n_random: int = 100  # number of random networks for comparison
    seed: int | None = None
Returns:
    dict[str, float]
    {"sigma": ..., "C_ratio": ..., "L_ratio": ..., "C_observed": ..., "C_random": ..., "L_observed": ..., "L_random": ...}
Small-worldness analysis
```

**Visualization 11**:
- Three network types side by side:
  - Regular lattice (high C, high L)
  - Random (low C, low L)
  - Small-world (high C, low L)
- Table showing C and L for each
- Caption: "Small-world networks combine advantages"

**Visualization 12**:
- Small-worldness analysis result
- Bar chart: C_observed/C_random and L_observed/L_random
- σ value annotated
- Caption: "Quantifying small-worldness"

---

## Section 8: Hub Detection

**Purpose**: Identify important nodes.

**Content to cover**:
- **Hubs**: highly connected, important nodes
- Multiple ways to define "importance":
  - **Degree centrality**: most connections
  - **Strength centrality**: strongest connections
  - **Betweenness centrality**: most shortest paths pass through
  - **Eigenvector centrality**: connected to other important nodes
- **Betweenness centrality**:
  - B_i = fraction of all shortest paths that pass through node i
  - High B = "bridge" node, critical for communication
- **Hub score** (composite):
  - Combine multiple centrality measures
  - Nodes ranking high on multiple measures = robust hubs
- **For brain networks**:
  - Hubs are critical for integration
  - Damage to hubs has outsized impact
  - Rich-club organization: hubs connected to each other

**Code to produce**:

Function 17: `compute_betweenness_centrality`
```
Parameters:
    adjacency: NDArray[np.float64]
Returns:
    NDArray[np.float64]
Betweenness centrality for each node
Normalized by total number of paths
```

Function 18: `compute_eigenvector_centrality`
```
Parameters:
    adjacency: NDArray[np.float64]
    max_iter: int = 100
Returns:
    NDArray[np.float64]
Eigenvector centrality (leading eigenvector of adjacency)
```

Function 19: `identify_hubs`
```
Parameters:
    adjacency: NDArray[np.float64]
    weighted_matrix: NDArray[np.float64] | None = None
    threshold_percentile: float = 90
Returns:
    dict[str, Any]
    {"hub_indices": ..., "degree": ..., "strength": ..., "betweenness": ..., "composite_score": ...}
Identify hubs based on multiple centrality measures
Hubs = nodes above threshold_percentile on composite score
```

**Visualization 13**:
- Network with nodes sized by betweenness centrality
- Highlight nodes that appear on many shortest paths
- Show example paths passing through hub
- Caption: "Betweenness centrality identifies bridge nodes"

**Visualization 14**:
- Topographic EEG layout
- Nodes colored by hub score
- Identify hub electrodes
- Caption: "Hub detection in EEG connectivity network"

---

## Section 9: Network Visualization

**Purpose**: Effective visual communication.

**Content to cover**:
- **Layout algorithms**:
  - Circular: nodes on circle, see all edges
  - Spring/force-directed: connected nodes attract
  - Topographic: use actual electrode positions
- **Visual encodings**:
  - Node size: degree or strength
  - Node color: clustering, module membership, centrality
  - Edge width: connection strength
  - Edge color: connection type (within vs between region)
- **Decluttering**:
  - Threshold weak edges
  - Use transparency for less important elements
  - Focus on top connections
- **For EEG**:
  - Topographic layout most interpretable
  - Can overlay on scalp map

**Code to produce**:

Function 20: `plot_network_circular`
```
Parameters:
    adjacency: NDArray[np.float64]
    node_labels: list[str] | None = None
    node_sizes: NDArray[np.float64] | None = None
    node_colors: NDArray[np.float64] | None = None
    edge_threshold: float = 0.0
    ax: matplotlib.axes.Axes | None = None
Returns:
    matplotlib.axes.Axes
Circular network layout
```

Function 21: `plot_network_spring`
```
Parameters:
    adjacency: NDArray[np.float64]
    node_sizes: NDArray[np.float64] | None = None
    node_colors: NDArray[np.float64] | None = None
    edge_threshold: float = 0.0
    seed: int | None = None
    ax: matplotlib.axes.Axes | None = None
Returns:
    matplotlib.axes.Axes
Force-directed (spring) layout
```

Function 22: `plot_network_topographic`
```
Parameters:
    adjacency: NDArray[np.float64]
    channel_positions: NDArray[np.float64]  # (n_channels, 2) x,y positions
    node_sizes: NDArray[np.float64] | None = None
    node_colors: NDArray[np.float64] | None = None
    edge_threshold: float = 0.0
    ax: matplotlib.axes.Axes | None = None
Returns:
    matplotlib.axes.Axes
Network overlaid on scalp topography
```

**Visualization 15**:
- Same network in three layouts:
  - Circular
  - Spring
  - Topographic (if positions available)
- Compare visual clarity
- Caption: "Network layout comparison"

---

## Section 10: Graph Metrics for EEG Connectivity

**Purpose**: Apply all concepts to EEG-like data.

**Content to cover**:
- **Full pipeline**:
  1. Compute connectivity matrix (PLV, coherence, etc.)
  2. Threshold to desired density
  3. Compute graph metrics
  4. Visualize network
- **Typical findings in EEG**:
  - Small-world organization
  - Posterior hubs (occipital, parietal)
  - Task-related changes in efficiency
  - Frequency-band specific networks
- **Compare conditions**:
  - Rest vs task
  - Eyes open vs closed
  - Different cognitive loads

**Code to produce**:

Function 23: `compute_graph_summary`
```
Parameters:
    connectivity_matrix: NDArray[np.float64]
    density: float = 0.2
Returns:
    dict[str, Any]
    {
        "density": ...,
        "mean_strength": ...,
        "global_clustering": ...,
        "global_efficiency": ...,
        "characteristic_path_length": ...,
        "small_worldness": ...,
        "hub_indices": ...,
        "degree": ...,
        "strength": ...,
        "clustering": ...,
        "betweenness": ...
    }
Complete graph analysis summary
```

**Visualization 16**:
- Full EEG connectivity analysis:
  - Panel A: Connectivity matrix
  - Panel B: Thresholded binary matrix
  - Panel C: Network visualization (topographic)
  - Panel D: Summary metrics bar chart
- Caption: "Complete EEG graph analysis pipeline"

---

## Section 11: Graphs for Hyperscanning

**Purpose**: THE key application — dual-brain networks!

**Content to cover**:
- **Hyperscanning networks**:
  - Nodes = channels from BOTH participants
  - Total nodes = n_channels × 2
  - Edges can be:
    - Within-P1 (internal to participant 1)
    - Within-P2 (internal to participant 2)
    - Between (inter-brain connections)
- **Network structure**:
  - Two "communities" (one per brain)
  - Between-brain edges = social coupling
- **Graph metrics for hyperscanning**:
  - Global metrics on combined network
  - Separate metrics for within vs between
  - Inter-brain efficiency: efficiency using only between edges
  - Participation coefficient: how much each node connects to other brain
- **Research questions**:
  - Do inter-brain connections increase during cooperation?
  - Are there "social hubs" bridging the two brains?
  - Does inter-brain network predict task performance?

**Code to produce**:

Function 24: `create_hyperscanning_graph`
```
Parameters:
    connectivity_p1: NDArray[np.float64]  # within-P1
    connectivity_p2: NDArray[np.float64]  # within-P2
    connectivity_between: NDArray[np.float64]  # P1 to P2
Returns:
    NDArray[np.float64]
Combined (2n × 2n) adjacency matrix
Properly structured with blocks
```

Function 25: `compute_hyperscanning_graph_metrics`
```
Parameters:
    hyperscanning_adjacency: NDArray[np.float64]
    n_channels_p1: int
Returns:
    dict[str, Any]
    {
        "global_efficiency": ...,
        "global_clustering": ...,
        "within_p1_efficiency": ...,
        "within_p2_efficiency": ...,
        "between_efficiency": ...,
        "inter_brain_density": ...,
        "participation_coefficient_p1": ...,  # per node
        "participation_coefficient_p2": ...,
        "social_hubs_p1": ...,  # nodes with high between-brain connectivity
        "social_hubs_p2": ...
    }
```

Function 26: `compute_participation_coefficient`
```
Parameters:
    adjacency: NDArray[np.float64]
    community_labels: NDArray[np.int64]  # community membership for each node
Returns:
    NDArray[np.float64]
Participation coefficient: how distributed are each node's connections across communities
High P_i = node connects to many communities
For hyperscanning: communities = the two participants
```

**Visualization 17**:
- Hyperscanning network visualization:
  - Two semicircles of nodes (P1 left, P2 right)
  - Within-brain edges in participant colors
  - Between-brain edges in distinct color (purple?)
  - Node sizes by participation coefficient
- Caption: "Dual-brain network in hyperscanning"

**Visualization 18**:
- Comparison bar chart:
  - Within-P1 efficiency
  - Within-P2 efficiency
  - Between-brain efficiency
- Show how inter-brain coupling compares to intra-brain
- Caption: "Intra-brain vs inter-brain network efficiency"

---

## Section 12: Hands-On Exercises

**Purpose**: Reinforce learning through practice.

**Exercise 1**: Matrix to Graph
- Create a simple 6×6 connectivity matrix
- Threshold at different values (0.3, 0.5, 0.7)
- Compute density for each
- Visualize resulting networks
- How does threshold affect network structure?

**Exercise 2**: Node Metrics
- Generate a network with clear hub structure
- Compute degree, strength, clustering for each node
- Identify the hub(s) using different criteria
- Do all metrics agree on hub identity?

**Exercise 3**: Path Length and Efficiency
- Create two networks: one dense, one sparse
- Compute characteristic path length and efficiency for each
- How do these metrics relate?
- What happens to efficiency as you remove edges?

**Exercise 4**: Small-World Analysis
- Generate random, lattice, and small-world networks
- Compute C and L for each
- Calculate small-worldness σ for the small-world network
- Verify it's > 1

**Exercise 5**: EEG-like Analysis
- Create a simulated 32-channel connectivity matrix
- Apply full pipeline: threshold → graph metrics → visualization
- Identify "hub" electrodes
- Is the network small-world?

**Exercise 6**: Hyperscanning Network
- Create connectivity for two "participants" (16 channels each)
- Combine into hyperscanning network
- Compare within-brain vs between-brain metrics
- Identify "social hub" electrodes (high between-brain connectivity)
- Visualize the dual-brain network

---

## Section 13: Functions for src/graph.py

**Purpose**: Create new graph analysis module.

**Create new file src/graph.py with all graph functions**:
- `threshold_matrix_absolute`
- `threshold_matrix_proportional`
- `get_graph_density`
- `compute_degree`
- `compute_strength`
- `compute_in_out_degree`
- `compute_clustering_coefficient`
- `compute_clustering_weighted`
- `compute_global_clustering`
- `compute_shortest_paths`
- `compute_characteristic_path_length`
- `compute_global_efficiency`
- `compute_local_efficiency`
- `generate_random_graph`
- `generate_lattice_graph`
- `compute_small_worldness`
- `compute_betweenness_centrality`
- `compute_eigenvector_centrality`
- `identify_hubs`
- `plot_network_circular`
- `plot_network_spring`
- `plot_network_topographic`
- `compute_graph_summary`
- `create_hyperscanning_graph`
- `compute_hyperscanning_graph_metrics`
- `compute_participation_coefficient`

---

## Summary

**Key takeaways to list**:
- Connectivity matrices = weighted adjacency matrices = graphs
- Thresholding converts weighted → binary graphs
- **Node metrics**: degree (connections), strength (weighted), clustering (local triangles)
- **Global metrics**: path length (integration), efficiency (harmonic paths), density
- **Small-world**: high clustering + short paths (typical for brain networks)
- **Hubs**: high degree, strength, betweenness nodes (critical for communication)
- Visualization: circular, spring, topographic layouts
- **For hyperscanning**: combine two brains into one graph
- Inter-brain edges reveal social coupling
- Participation coefficient identifies "social hubs"
- Graph metrics summarize complex connectivity patterns into interpretable numbers

---

## Discussion Questions

1. You threshold a connectivity matrix at different densities (10%, 20%, 30%) and get different hub locations. Which result should you trust? How would you report this?

2. A patient population shows lower global efficiency than controls. What does this mean functionally? What cognitive effects might you predict?

3. During hyperscanning, one participant has much higher participation coefficient (more between-brain connections) than the other. What might this indicate about their role in the interaction?

4. Your EEG network has σ = 3.2 (small-world). A colleague asks "so what?" — how would you explain why this matters for brain function?

5. You want to compare networks across participants, but they have different numbers of significant connections. How would proportional thresholding help? What are its limitations?

---

## Visualization Checklist

| # | Type | Description | Colors |
|---|------|-------------|--------|
| 1 | Graph + matrix | Simple graph example | signal_1 |
| 2 | 2×2 grid | Graph types | signal_1, signal_2 |
| 3 | Pipeline | Matrix → threshold → graph | signal_1 |
| 4 | 3 graphs | Threshold effect | signal_1 |
| 5 | Network | Nodes sized by degree | signal_1 |
| 6 | Histogram | Degree distributions | signal_1, signal_2 |
| 7 | 2 networks | Clustering comparison | highlight |
| 8 | Topographic | Clustering coefficient | colormap |
| 9 | Network | Shortest path highlighted | signal_1, highlight |
| 10 | 2 networks | Efficiency comparison | colormap |
| 11 | 3 networks | Regular, random, small-world | signal_1 |
| 12 | Bar chart | Small-worldness analysis | signal_1 |
| 13 | Network | Betweenness centrality | colormap |
| 14 | Topographic | Hub detection | colormap |
| 15 | 3 layouts | Circular, spring, topographic | signal_1 |
| 16 | 4-panel | Complete EEG pipeline | signal_1 |
| 17 | Dual network | Hyperscanning visualization | subject_1, subject_2 |
| 18 | Bar chart | Intra vs inter-brain efficiency | subject_1, subject_2 |

---

## Code Quality Checklist

- [ ] All functions have complete type hints
- [ ] All functions have NumPy-style docstrings
- [ ] No magic numbers (use named constants)
- [ ] Figures follow STYLE_GUIDE.md specifications
- [ ] Code is PEP 8 compliant
- [ ] Functions over 20 lines flagged for src/ extraction
- [ ] Shortest path algorithm handles disconnected graphs
- [ ] Clustering coefficient handles low-degree nodes
- [ ] Threshold functions preserve matrix shape
- [ ] Visualization functions accept optional axes
- [ ] Hyperscanning functions handle different channel counts
- [ ] Network layouts are reproducible (seeded)