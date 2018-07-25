package hopshackle.DL4JUtilities;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.*;

public class TrainQFunction {

    private static int batchSize = 16;
    private static int hiddenNeurons = 20;
    private static double learningRate = 1e-5;
    private static int epochs = 200;
    private static double trainingPercentage = 1.0;
    private static double momentum = 0.9;
    private static double regularisationL1 = 0.0;
    private static double regularisationL2 = 1e-6;
    private static Activation hiddenActivation = Activation.RELU;
    private static Activation outputActivation = Activation.RECTIFIEDTANH;

    /*
    Takes a file as an argument, and then uses this to train a simple Neural Network
    which is written to a second location as a file

    We assume that the first column in the file is the target value
     */
    public static void main(String[] args) {
        if (args.length < 3) throw new AssertionError("Need three arguments for input and output locations, plus number of categories");
        String inputLocation = args[0];
        String outputLocation = args[1];
        int numberOfRules = Integer.valueOf(args[2]);
        if (args.length > 3) epochs = Integer.valueOf(args[3]);
        if (args.length > 4) batchSize = Integer.valueOf(args[4]);
        if (args.length > 5) momentum = Double.valueOf(args[5]);
        if (args.length > 6) learningRate = Double.valueOf(args[6]);
        if (args.length > 7) hiddenNeurons = Integer.valueOf(args[7]);
        if (args.length > 8) regularisationL1 = Double.valueOf(args[8]);
        if (args.length > 9) regularisationL2 = Double.valueOf(args[9]);
        if (args.length > 10) hiddenActivation = TrainEvalFunction.getActivation(args[10], hiddenActivation);
        if (args.length > 11) outputActivation = TrainEvalFunction.getActivation(args[11], outputActivation);
        if (args.length > 12) trainingPercentage = Double.valueOf(args[12]);

        RecordReader recordReader = new CSVRecordReader('\t');
        try {
            recordReader.initialize(new FileSplit(new File(inputLocation)));
        } catch (Exception e) {
            throw new AssertionError("Error processing file " + inputLocation + ":\n" + e.toString());
        }

        System.out.println("Starting...");
        List<List<Writable>> allData = new ArrayList<>();
        while (recordReader.hasNext())
            allData.add(recordReader.next());

        List<List<Writable>> trainData = allData.subList(0, (int) (allData.size() * trainingPercentage));
        List<List<Writable>> testData = allData.subList((int) (allData.size() * trainingPercentage), allData.size());
        Collections.shuffle(trainData);

        CollectionRecordReader crrTrain = new CollectionRecordReader(trainData);
        CollectionRecordReader crrTest = new CollectionRecordReader(testData);

        //     DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, 0, numberOfRules - 1, true);
        DataSetIterator iterator = new RecordReaderDataSetIterator(crrTrain, batchSize, 0, numberOfRules - 1, true);

        NormalizerStandardize normalizer = new NormalizerStandardize();
        normalizer.fit(iterator);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        iterator.setPreProcessor(normalizer); // then set this to pre-process the data

        DataSetIterator testIterator = new RecordReaderDataSetIterator(crrTest, testData.size(), 0, numberOfRules - 1, true);
        testIterator.setPreProcessor(normalizer); // then set this to pre-process the test data too!

        System.out.println("Completed pre-processing...");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .l1(regularisationL1)
                .l2(regularisationL2)
                .updater(new Nesterovs(learningRate, momentum))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(iterator.inputColumns()).nOut(hiddenNeurons)
                        .activation(hiddenActivation)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                        .activation(outputActivation)
                        .nIn(hiddenNeurons)
                        .nOut(numberOfRules)
                        .build())
                .pretrain(false).backprop(true).build();


        System.out.println("Building model...");
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        DataSet testDataSet = testData.isEmpty() ? null : ((RecordReaderDataSetIterator) testIterator).next();
        if (testData.size() > 0)
            System.out.println(String.format("Before training the test error is %.3f", model.score(testDataSet)));
        for (int n = 0; n < epochs; n++) {
            model.fit(iterator);
            double testScore = (testDataSet != null) ? model.score(testDataSet) : Double.NaN;
            System.out.println(String.format("Epoch %3d has error %.4f, and test error %.4f", n, model.score(), testScore));
        }

        //Save the model
        boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        try {
            ModelSerializer.writeModel(model, outputLocation, saveUpdater);
        } catch (Exception e) {
            throw new AssertionError("Error writing file " + outputLocation + ":\n" + e.toString());
        }

        // Now we want to save the normalizer to a binary file. For doing this, one can use the NormalizerSerializer.
        NormalizerSerializer serializer = NormalizerSerializer.getDefault();

        // Save the normalizer to a temporary file
        try {
            serializer.write(normalizer, outputLocation + ".normal");
        } catch (Exception e) {
            throw new AssertionError("Error writing file " + outputLocation + ".normal:\n" + e.toString());
        }

        HopshackleNN asHopshackle = new HopshackleNN(model, normalizer);
        asHopshackle.writeToFile(outputLocation.replaceAll(".model", ".params"));
    }
}
