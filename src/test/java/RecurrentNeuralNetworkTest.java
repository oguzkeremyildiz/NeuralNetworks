import NeuralNetworks.ActivationFunction.Activation;
import NeuralNetworks.Initializer.Initializer;
import NeuralNetworks.InstanceList.VectorizedInstanceList;
import NeuralNetworks.Net.RecurrentNeuralNetwork;
import Util.FileUtils;
import org.junit.Test;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.LinkedList;
import Math.*;

import static org.junit.Assert.*;

public class RecurrentNeuralNetworkTest {

    @Test
    public void testShallowParse() throws MatrixDimensionMismatch, MatrixRowColumnMismatch, VectorSizeMismatch {
        VectorizedInstanceList trainList = null, testList = null;
        ObjectInputStream outObject;
        try {
            outObject = new ObjectInputStream(FileUtils.getInputStream("src/main/resources/Dataset/ShallowParse/shallow-list-1-train.bin"));
            trainList = (VectorizedInstanceList) outObject.readObject();
            outObject = new ObjectInputStream(FileUtils.getInputStream("src/main/resources/Dataset/ShallowParse/shallow-list-1-test.bin"));
            testList = (VectorizedInstanceList) outObject.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
        LinkedList<Integer> hiddenLayers = new LinkedList<>();
        LinkedList<Activation> activations = new LinkedList<>();
        hiddenLayers.add(10);
        activations.add(Activation.TANH);
        RecurrentNeuralNetwork net = new RecurrentNeuralNetwork(1, hiddenLayers, trainList, activations, Initializer.RANDOM);
        net.train(100, 0.2, 0.99, 0.0);
        if (testList != null) {
            double accuracy = net.test(testList);
            assertEquals(54.694192823537676, accuracy, 0.01);
        }
    }

    @Test
    public void testNER() throws MatrixDimensionMismatch, MatrixRowColumnMismatch, VectorSizeMismatch {
        VectorizedInstanceList trainList = null, testList = null;
        ObjectInputStream outObject;
        try {
            outObject = new ObjectInputStream(FileUtils.getInputStream("src/main/resources/Dataset/NER/ner-list-1-train.bin"));
            trainList = (VectorizedInstanceList) outObject.readObject();
            outObject = new ObjectInputStream(FileUtils.getInputStream("src/main/resources/Dataset/NER/ner-list-1-test.bin"));
            testList = (VectorizedInstanceList) outObject.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
        LinkedList<Integer> hiddenLayers = new LinkedList<>();
        LinkedList<Activation> activations = new LinkedList<>();
        hiddenLayers.add(15);
        hiddenLayers.add(15);
        activations.add(Activation.TANH);
        activations.add(Activation.TANH);
        RecurrentNeuralNetwork net = new RecurrentNeuralNetwork(1, hiddenLayers, trainList, activations, Initializer.RANDOM);
        net.train(100, 0.01, 0.99, 0.5);
        if (testList != null) {
            double accuracy = net.test(testList);
            assertEquals(91.5736754239522, accuracy, 0.01);
        }
    }

    @Test
    public void testSentiment() throws MatrixRowColumnMismatch, MatrixDimensionMismatch, VectorSizeMismatch {
        VectorizedInstanceList trainList = null, testList = null;
        ObjectInputStream outObject;
        try {
            outObject = new ObjectInputStream(FileUtils.getInputStream("src/main/resources/Dataset/SentimentAnalysis/sentiment-list-2-train.bin"));
            trainList = (VectorizedInstanceList) outObject.readObject();
            outObject = new ObjectInputStream(FileUtils.getInputStream("src/main/resources/Dataset/SentimentAnalysis/sentiment-list-2-test.bin"));
            testList = (VectorizedInstanceList) outObject.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
        LinkedList<Integer> hiddenLayers = new LinkedList<>();
        LinkedList<Activation> activations = new LinkedList<>();
        hiddenLayers.add(30);
        hiddenLayers.add(20);
        activations.add(Activation.TANH);
        activations.add(Activation.TANH);
        RecurrentNeuralNetwork net = new RecurrentNeuralNetwork(1, hiddenLayers, trainList, activations, Initializer.RANDOM);
        net.train(100, 0.01, 0.99, 0.5);
        if (testList != null) {
            double accuracy = net.test(testList);
            assertEquals(78.72340425531915, accuracy, 0.01);
        }
    }
}
