package hopshackle.DL4JUtilities;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.StandardizeSerializerStrategy;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

public class EvaluateWithModel {


    public static void main(String[] args) {

        // read in data file
        INDArray rawData;
        List<List<Double>> X = new ArrayList<>();
        List<Double> Y = new ArrayList<>();
        try {
            FileReader reader = new FileReader("C:/simulation/hanabi/PowerPlantData.txt");
            BufferedReader bufferedReader = new BufferedReader(reader);
            do {
                String[] d = bufferedReader.readLine().split("\\t");
                List<Double> data = Arrays.stream(d).map(Double::valueOf).collect(Collectors.toList());
                Y.add(data.get(0));
                data.remove(0);
                X.add(data);
            } while (true);
        } catch (Exception e){
            e.printStackTrace();
        }

        for (int i = 0; i < Y.size(); i++) {
    //        System.out.println(String.format("%.1f <- %s", Y.get(i), X.get(i)));
        }
        // read in model and normaliser
        MultiLayerNetwork model = null;
        NormalizerStandardize normalizer = null;
        try {
            model = ModelSerializer.restoreMultiLayerNetwork("C:/Simulation/hanabi/PowerPlant.model");
            NormalizerSerializer ns = new NormalizerSerializer();
            ns.addStrategy(new StandardizeSerializerStrategy());
            normalizer = ns.restore("C:/Simulation/hanabi/PowerPlant.model.normal");
        } catch (Exception e) {
            System.out.println("Error when reading in Model");
            e.printStackTrace();
        }
        // process file and output actual versus expected values

        for (int i = 0; i < X.size(); i++) {
            double[][] temp = new double[1][X.get(i).size()];
            temp[0] = X.get(i).stream().mapToDouble(x -> x).toArray();
            INDArray input = new NDArray(temp);
            normalizer.transform(input);
            double estY = model.output(input).getDouble(0);
            System.out.println(String.format("Line %3d\tY: %.2f\tY': %.2f", i+1, Y.get(i), estY));
        }

        double[] params = model.params().toDoubleVector();
        double[] means = normalizer.getMean().toDoubleVector();
        double[] std = normalizer.getStd().toDoubleVector();
        System.out.println("Params: " + Arrays.stream(params).mapToObj(d -> String.format("%.4f", d)).collect(Collectors.joining("\t")));
        System.out.println("Means: " + Arrays.stream(means).mapToObj(d -> String.format("%.4f", d)).collect(Collectors.joining("\t")));
        System.out.println("StDev: " + Arrays.stream(std).mapToObj(d -> String.format("%.4f", d)).collect(Collectors.joining("\t")));

        HopshackleNN test = new HopshackleNN(model, normalizer);
        test.writeToFile("C://simulation//hanabi//ConvertedDL4JModel.params");
    }
}
