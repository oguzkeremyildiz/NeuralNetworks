import NeuralNetworks.ActivationFunction.Activation;
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
    public void testShallowParse() throws MatrixDimensionMismatch, MatrixRowColumnMismatch {
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
        hiddenLayers.add(10);
        RecurrentNeuralNetwork net = new RecurrentNeuralNetwork(1, hiddenLayers, trainList, Activation.TANH);
        net.train(100, 0.2, 0.99, 0.0);
        if (testList != null) {
            double accuracy = net.test(testList);
            assertEquals(54.84165437820378, accuracy, 0.01);
        }
    }

    @Test
    public void testNER() throws MatrixDimensionMismatch, MatrixRowColumnMismatch {
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
        hiddenLayers.add(15);
        hiddenLayers.add(15);
        RecurrentNeuralNetwork net = new RecurrentNeuralNetwork(1, hiddenLayers, trainList, Activation.SIGMOID);
        net.train(100, 0.01, 0.99, 0.5);
        if (testList != null) {
            double accuracy = net.test(testList);
            assertEquals(92.29417450136192, accuracy, 0.01);
        }
    }

    @Test
    public void testSentiment() throws MatrixRowColumnMismatch, MatrixDimensionMismatch {
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
        hiddenLayers.add(30);
        hiddenLayers.add(20);
        RecurrentNeuralNetwork net = new RecurrentNeuralNetwork(1, hiddenLayers, trainList, Activation.TANH);
        net.train(100, 0.01, 0.99, 0.5);
        if (testList != null) {
            double accuracy = net.test(testList);
            assertEquals(78.0984481426866, accuracy, 0.01);
        }
    }
}
