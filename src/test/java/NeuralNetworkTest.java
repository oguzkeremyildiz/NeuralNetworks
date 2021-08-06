import ActivationFunctions.Activation;
import org.junit.Test;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.LinkedList;
import java.util.Scanner;

import static org.junit.Assert.*;

public class NeuralNetworkTest {

    @Test
    public void testTrain() throws InputSizeException, FileNotFoundException {
        InstanceList list = new InstanceList(new Scanner(new File("Dataset/dermatology.txt")), ",");
        LinkedList<Integer> hiddenLayers = new LinkedList<>();
        hiddenLayers.add(20);
        NeuralNetwork net = new NeuralNetwork(1, hiddenLayers, list, Activation.SIGMOID);
        net.train(1000, 0.01, 0.99);
        double accuracy = net.test(list);
        assertEquals(99.72677595628416, accuracy, 0.01);
        list = new InstanceList(new Scanner(new File("Dataset/bupa.txt")), ",");
        hiddenLayers = new LinkedList<>();
        hiddenLayers.add(15);
        hiddenLayers.add(15);
        net = new NeuralNetwork(1, hiddenLayers, list, Activation.SIGMOID);
        net.train(100, 0.01, 0.99);
        accuracy = net.test(list);
        assertEquals(73.91304347826087, accuracy, 0.01);
        list = new InstanceList(new Scanner(new File("Dataset/iris.txt")), ",");
        hiddenLayers = new LinkedList<>();
        hiddenLayers.add(5);
        hiddenLayers.add(5);
        net = new NeuralNetwork(1, hiddenLayers, list, Activation.SIGMOID);
        net.train(500, 0.1, 0.99);
        accuracy = net.test(list);
        assertEquals(98.66666666666667, accuracy, 0.01);
    }

}