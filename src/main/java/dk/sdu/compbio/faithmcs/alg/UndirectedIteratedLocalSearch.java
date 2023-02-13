package dk.sdu.compbio.faithmcs.alg;

import com.google.common.collect.Sets;
import dk.sdu.compbio.faithmcs.UndirectedAlignment;
import dk.sdu.compbio.faithmcs.UndirectedEdgeMatrix;
import dk.sdu.compbio.faithmcs.network.Edge;
import dk.sdu.compbio.faithmcs.network.UndirectedNetwork;
import dk.sdu.compbio.faithmcs.network.Node;
import org.jgrapht.alg.NeighborIndex;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class UndirectedIteratedLocalSearch implements IteratedLocalSearch {
    private final int n, M;
    private final List<UndirectedNetwork> networks;
    private final int min_lsi_swaps;
    private final int MIN_LSI_SWAP_RATIO = 1000;
    private float perturbation_amount;

    private final List<NeighborIndex<Node,Edge>> indices;
    private final List<List<Node>> nodes;
    private final UndirectedEdgeMatrix edges;
    private final int[][] best_positions;
    private int quality, best_quality;
    private final Random rand;

    public UndirectedIteratedLocalSearch(List<UndirectedNetwork> networks, float perturbation_amount) {
        this.networks = networks;
        // min_lsi_swaps uses the min of the edges instead of the max because the # of swaps is limited to the min # of edges
        this.min_lsi_swaps = (int)(networks.stream().mapToInt(g -> g.edgeSet().size()).min().getAsInt() / MIN_LSI_SWAP_RATIO);
        this.perturbation_amount = perturbation_amount;

        n = networks.size();
        M = networks.stream().mapToInt(g -> g.vertexSet().size()).max().getAsInt();

        indices = new ArrayList<>();

        int fid = 0;
        for(UndirectedNetwork network : networks) {
            while(network.vertexSet().size() < M) {
                Node fake_node = new Node("$fake$" + fid++, true);
                network.addVertex(fake_node);
            }

            indices.add(new NeighborIndex<>(network));
        }

        nodes = networks.stream()
                .map(network -> new ArrayList<>(network.vertexSet()))
                .collect(Collectors.toList());

        for(int i = 0; i < n; ++i) {
            nodes.get(i).sort(Comparator.comparingInt(networks.get(i)::degreeOf).reversed());
            int pos = 0;
            for(Node node : nodes.get(i)) {
                node.setPosition(pos++);
            }
        }

        edges = new UndirectedEdgeMatrix(networks);
        rand = new Random();

        best_positions = new int[n][M];
        copyPositions(nodes, best_positions);
        best_quality = edges.countEdges();
    }

    @Override
    public void run(int max_nonimproving, int max_num_steps) {
        int nonimproving = 0;
        int num_steps = 0;
        while(nonimproving < max_nonimproving && num_steps < max_num_steps) {
            nonimproving++;
            if(step()) {
                nonimproving = 0;
            }
            num_steps++;
            System.err.println(String.format("step: %d, current: %d edges, best: %d edges", num_steps, quality, best_quality));
        }
    }

    @Override
    public boolean step() {
        // perturbation step
        int count = Math.round(M * perturbation_amount);
        for(int i = 1; i < n; ++i) {
            for(int rep = 0; rep < count; ++rep) {
                int j = rand.nextInt(M);
                int k;
                do k = rand.nextInt(M); while(k == j);
                swap(indices.get(i), nodes.get(i).get(j), nodes.get(i).get(k));
            }
        }

        // local search step
        boolean repeat = true;
        int num_iterations = 0;
        int num_swaps_this_iteration = 0;
        while(repeat) {
            repeat = false;
            num_swaps_this_iteration = 0;
            long iteration_start_time = System.currentTimeMillis();
            for (int i = 1; i < n; ++i) {
                for (int j = 0; j < M-1; ++j) {
                    int finalI = i;
                    int finalJ = j;

                    List<Integer> dts = IntStream.range(j+1, M)
                            .parallel()
                            .mapToObj(k -> delta(indices.get(finalI), nodes.get(finalI).get(finalJ), nodes.get(finalI).get(k)))
                            .collect(Collectors.toList());

                    Integer best = IntStream.range(j+1, M)
                            .parallel()
                            .boxed()
                            .max(Comparator.comparingInt(k -> dts.get(k-(finalJ+1)))).get();

                    int dt = dts.get(best-(j+1));

                    if(dt > 0) {
                        num_swaps_this_iteration += 1;
                        swap(indices.get(i), nodes.get(i).get(j), nodes.get(i).get(best));
                    }
                }
            }
            num_iterations += 1;
            long iteration_end_time = System.currentTimeMillis();

            if (num_swaps_this_iteration >= this.min_lsi_swaps) {
                repeat = true;
            }

            System.err.println("LSI " + num_iterations + ", S=" + num_swaps_this_iteration);
            System.err.println("LSI " + num_iterations + " took " + (iteration_end_time - iteration_start_time) + "ms");
        }

        // count edges
        quality = edges.countEdges();
        if(quality > best_quality) {
            best_quality = quality;
            copyPositions(nodes, best_positions);
            return true;
        }
        return false;
    }

    private void copyPositions(List<List<Node>> nodes, int[][] positions) {
        for(int i = 0; i < nodes.size(); ++i) {
            for(int j = 0; j < nodes.get(i).size(); ++j) {
                positions[i][j] = nodes.get(i).get(j).getPosition();
            }
        }
    }

    private int delta(NeighborIndex<Node,Edge> index, Node u, Node v) {
        int delta = 0;

        int i = u.getPosition();
        int j = v.getPosition();

        for(Node w : Sets.difference(index.neighborsOf(u), index.neighborsOf(v))){
            if(w != v) {
                int l = w.getPosition();
                delta -= 2 * edges.get(i, l) - 1;
                delta += 2 * edges.get(j, l) + 1;
            }
        }

        for(Node w : Sets.difference(index.neighborsOf(v), index.neighborsOf(u))) {
            if(w != u) {
                int l = w.getPosition();
                delta -= 2 * edges.get(j, l) - 1;
                delta += 2 * edges.get(i, l) + 1;
            }
        }

        return delta;
    }

    private void swap(NeighborIndex<Node,Edge> index, Node u, Node v) {
        int i = u.getPosition();
        int j = v.getPosition();

        for(Node w : Sets.difference(index.neighborsOf(u), index.neighborsOf(v))) {
            if(w != v) {
                int l = w.getPosition();
                edges.decrement(i, l);
                edges.increment(j, l);
            }
        }

        for(Node w : Sets.difference(index.neighborsOf(v), index.neighborsOf(u))) {
            if(w != u) {
                int l = w.getPosition();
                edges.decrement(j, l);
                edges.increment(i, l);
            }
        }

        u.setPosition(j);
        v.setPosition(i);
    }

    @Override
    public UndirectedAlignment getAlignment() {
        // copy best solution back into nodes
        for(int i = 0; i < n; ++i) {
            for(int j = 0; j < M; ++j) {
                nodes.get(i).get(j).setPosition(best_positions[i][j]);
            }
        }

        // Sort nodes on position to obtain alignment
        for(List<Node> node_list : nodes) {
            node_list.sort(Comparator.comparingInt(Node::getPosition));
        }

        return new UndirectedAlignment(nodes, networks);
    }

    @Override
    public int getCurrentNumberOfEdges() {
        return quality;
    }

    @Override
    public int getBestNumberOfEdges() {
        return best_quality;
    }

    @Override
    public void setPerturbationAmount(float a) {
        this.perturbation_amount = a;
    }
}
