package dk.sdu.compbio.netgale;

import dk.sdu.compbio.netgale.alg.Aligner;
import dk.sdu.compbio.netgale.alg.LocalSearch;
import dk.sdu.compbio.netgale.network.Network;
import dk.sdu.compbio.netgale.network.io.ImportException;
import dk.sdu.compbio.netgale.network.io.NetworkReader;
import dk.sdu.compbio.netgale.network.io.NetworkWriter;
import dk.sdu.compbio.netgale.network.Node;
import org.apache.commons.cli.*;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class JNetGALE {
    public static void main(String[] args) throws ParseException, FileNotFoundException, ImportException {
        Option iterations_option = Option.builder("i").longOpt("iterations").hasArg().build();
        Option output_option = Option.builder("o").longOpt("output").hasArg().build();
        Option output_graph_option = Option.builder("O").longOpt("write-network").hasArg().build();

        Options options = new Options();
        options.addOption(iterations_option);
        options.addOption(output_option);
        options.addOption(output_graph_option);

        CommandLineParser parser = new DefaultParser();
        CommandLine cmd = parser.parse(options, args);

        if(cmd.getArgList().size() < 2) {
            System.err.println("error: Needs at least two networks for alignment.");
            System.exit(1);
        }

        int iterations = Integer.parseInt(cmd.getOptionValue("i", "20"));

        List<Network> networks = new ArrayList<>();
        for(String path : cmd.getArgList()) {
            Network network = new Network();
            NetworkReader.read(network, new File(path));
            networks.add(network);
        }

        Model model = new Model();

        Aligner aligner = new LocalSearch(networks, model);
        aligner.run(iterations);
        Alignment alignment = aligner.getAlignment();

        if(cmd.hasOption("o")) {
            writeAlignment(alignment, new File(cmd.getOptionValue("o")));
        }

        if(cmd.hasOption("O")) {
            NetworkWriter.write(alignment.buildNetwork(), new File(cmd.getOptionValue("O")));
        }
    }

    private static void writeAlignment(Alignment alignment, File file) throws FileNotFoundException {
        PrintWriter pw = new PrintWriter(file);
        List<List<Node>> align = alignment.getAlignment();
        int n = align.size();
        int M = align.get(0).size();

        for(int j = 0; j < M; ++j) {
            int finalJ = j;
            pw.println(IntStream.range(0, n)
                    .mapToObj(i -> align.get(i).get(finalJ))
                    .filter(node -> !node.isFake())
                    .map(Node::toString)
                    .collect(Collectors.joining("\t"))
            );
        }
        pw.close();
    }
}
