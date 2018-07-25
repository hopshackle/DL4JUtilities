package hopshackle.DL4JUtilities;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import com.google.gson.*;

import java.io.*;
import java.util.Arrays;
import java.util.stream.Collectors;

public class HopshackleNN {

    enum ACTIVATION {
        RELU, RECTIFIED_TANH, SOFTMAX
    }

    private ACTIVATION[] layers;
    private int[] inputsByLayer;
    private int[] outputsByLayer;
    private double[][][] weightsByLayerNeuronAndInput;
    private double[][] biasByLayerAndNeuron;

    private double[] meansForInput;
    private double[] stdForInput;

    public HopshackleNN(MultiLayerNetwork model, NormalizerStandardize normaliser) {
        meansForInput = normaliser.getMean().toDoubleVector();
        stdForInput = normaliser.getStd().toDoubleVector();

        int numberOfLayers = model.getLayers().length;
        layers = new ACTIVATION[numberOfLayers];
        inputsByLayer = new int[numberOfLayers];
        outputsByLayer = new int[numberOfLayers];
        layers = new ACTIVATION[]{ACTIVATION.RELU, ACTIVATION.RECTIFIED_TANH};
        // hard-coded for the moment, as there is no easy way to pull this from the model
        weightsByLayerNeuronAndInput = new double[numberOfLayers][][];
        biasByLayerAndNeuron = new double[numberOfLayers][];

        for (int layer = 0; layer < numberOfLayers; layer++) {
            inputsByLayer[layer] = model.layerInputSize(layer);
            outputsByLayer[layer] = model.layerSize(layer);
            weightsByLayerNeuronAndInput[layer] = new double[outputsByLayer[layer]][inputsByLayer[layer]];
            biasByLayerAndNeuron[layer] = new double[outputsByLayer[layer]];
        }
        double[] allParams = model.params().toDoubleVector();
        int next = 0;
        for (int layer = 0; layer < numberOfLayers; layer++) {
            for (int neuron = 0; neuron < outputsByLayer[layer]; neuron++) {
                for (int input = 0; input < inputsByLayer[layer]; input++) {
                    weightsByLayerNeuronAndInput[layer][neuron][input] = allParams[next];
                    next++;
                }
            }
            for (int neuron = 0; neuron < outputsByLayer[layer]; neuron++) {
                biasByLayerAndNeuron[layer][neuron] = allParams[next];
                next++;
            }
        }
    }


    public static HopshackleNN createFromStream(InputStream input) {
        Gson gson = new Gson();
        String asString = new BufferedReader(new InputStreamReader(input))
                .lines().collect(Collectors.joining("\n"));
        return gson.fromJson(asString, HopshackleNN.class);
    }

    public void writeToFile(String fileLocation) {
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        String asGSON = gson.toJson(this);
        try {
            FileWriter writer = new FileWriter(new File(fileLocation));
            writer.write(asGSON);
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    public double[] process(double[] data) {
        if (data.length != inputsByLayer[0])
            throw new AssertionError("Must have " + inputsByLayer[0] + " inputs instead of " + data.length);

        double[] input = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            input[i] = (data[i] - meansForInput[i]) / stdForInput[i];
        }
        double[][] activationsByLayerAndNeuron = new double[layers.length][];
        for (int layer = 0; layer < layers.length; layer++) {
            activationsByLayerAndNeuron[layer] = new double[outputsByLayer[layer]];
            for (int neuron = 0; neuron < outputsByLayer[layer]; neuron++) {
                double activation = 0.0;
                for (int previousOutput = 0; previousOutput < input.length; previousOutput++) {
                    activation += input[previousOutput] * weightsByLayerNeuronAndInput[layer][neuron][previousOutput];
                }
                activation += biasByLayerAndNeuron[layer][neuron];
                switch (layers[layer]) {
                    case RELU:
                        activation = Math.max(0, activation);
                        break;
                    case RECTIFIED_TANH:
                        activation = Math.max(0, Math.tanh(activation));
                        break;
                    case SOFTMAX:
                        activation = Math.exp(activation);
                        break;
                    default:
                }
                activationsByLayerAndNeuron[layer][neuron] = activation;
            }
            if (layers[layer] == ACTIVATION.SOFTMAX) {
                // whole layer is linked
                double totalActivation = Arrays.stream(activationsByLayerAndNeuron[layer]).sum();
                for (int i = 0; i < activationsByLayerAndNeuron[layer].length; i++) activationsByLayerAndNeuron[layer][i] /= totalActivation;
            }
            input = activationsByLayerAndNeuron[layer]; // the input to the next layer
        }
        return input;   // and we return the last activations
    }
}
