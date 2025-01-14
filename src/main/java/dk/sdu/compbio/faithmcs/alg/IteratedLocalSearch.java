package dk.sdu.compbio.faithmcs.alg;

import dk.sdu.compbio.faithmcs.Alignment;

public interface IteratedLocalSearch {
    boolean step();
    void run(int max_nonimproving, int max_num_steps);
    Alignment getAlignment();
    int getCurrentNumberOfEdges();
    int getBestNumberOfEdges();
    void setPerturbationAmount(float a);
}
