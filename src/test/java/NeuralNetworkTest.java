import NeuralNetworks.ActivationFunction.Activation;
import NeuralNetworks.Initializer.Initializer;
import NeuralNetworks.InputSizeException;
import NeuralNetworks.InstanceList.AttributeType;
import NeuralNetworks.InstanceList.InstanceList;
import NeuralNetworks.Net.NeuralNetwork;
import org.junit.Test;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.LinkedList;
import java.util.Scanner;
import Math.*;

import static org.junit.Assert.*;

public class NeuralNetworkTest {

    @Test
    public void testTrain() throws InputSizeException, FileNotFoundException, MatrixDimensionMismatch, MatrixRowColumnMismatch {
        InstanceList list = new InstanceList(new Scanner(new File("src/main/resources/Dataset/dermatology.txt")), ",", AttributeType.CONTINUOUS);
        LinkedList<Integer> hiddenLayers = new LinkedList<>();
        LinkedList<Activation> activations = new LinkedList<>();
        hiddenLayers.add(25);
        activations.add(Activation.SIGMOID);
        NeuralNetwork net = new NeuralNetwork(1, hiddenLayers, list, activations, Initializer.RANDOM);
        net.train(1000, 0.01, 0.99, 0.5);
        double accuracy = net.test(list);
        assertEquals(100.0, accuracy, 0.01);
        list = new InstanceList(new Scanner(new File("src/main/resources/Dataset/bupa.txt")), ",", AttributeType.CONTINUOUS);
        hiddenLayers = new LinkedList<>();
        activations = new LinkedList<>();
        hiddenLayers.add(15);
        hiddenLayers.add(15);
        activations.add(Activation.SIGMOID);
        activations.add(Activation.SIGMOID);
        net = new NeuralNetwork(1, hiddenLayers, list, activations, Initializer.RANDOM);
        net.train(200, 0.01, 0.99, 0.0);
        accuracy = net.test(list);
        assertEquals(77.10144927536231, accuracy, 0.01);
        list = new InstanceList(new Scanner(new File("src/main/resources/Dataset/iris.txt")), ",", AttributeType.CONTINUOUS);
        hiddenLayers = new LinkedList<>();
        activations = new LinkedList<>();
        hiddenLayers.add(5);
        hiddenLayers.add(5);
        activations.add(Activation.SIGMOID);
        activations.add(Activation.SIGMOID);
        net = new NeuralNetwork(1, hiddenLayers, list, activations, Initializer.RANDOM);
        net.train(500, 0.1, 0.99, 0.3);
        accuracy = net.test(list);
        assertEquals(99.33333333333333, accuracy, 0.01);
        list = new InstanceList(new Scanner(new File("src/main/resources/Dataset/5.txt")), " ", AttributeType.DISCRETE);
        hiddenLayers = new LinkedList<>();
        activations = new LinkedList<>();
        hiddenLayers.add(30);
        hiddenLayers.add(30);
        activations.add(Activation.SIGMOID);
        activations.add(Activation.SIGMOID);
        net = new NeuralNetwork(1, hiddenLayers, list, activations, Initializer.RANDOM);
        net.train(1000, 0.1, 0.99, 0.3);
        accuracy = net.test(list);
        assertEquals(88.62385321100918, accuracy, 0.01);
    }

}