package hopshackle.DL4JUtilities;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.StandardizeSerializerStrategy;

public class ConvertDL4JToHopshackle {

    public static void main(String[] args) {
        String location = args[0];
        String newLocation = args[1];

        // read in model and normaliser
        MultiLayerNetwork model = null;
        NormalizerStandardize normalizer = null;
        try {
            model = ModelSerializer.restoreMultiLayerNetwork(location);
            NormalizerSerializer ns = new NormalizerSerializer();
            ns.addStrategy(new StandardizeSerializerStrategy());
            normalizer = ns.restore(location + ".normal");
        } catch (Exception e) {
            System.out.println("Error when reading in Model");
            e.printStackTrace();
        }

        HopshackleNN asHopshackle = new HopshackleNN(model, normalizer);
        asHopshackle.writeToFile(newLocation);
    }
}
