import Corpus.Corpus;
import Dictionary.VectorizedDictionary;
import NeuralNetworks.ActivationFunctions.Activation;
import NeuralNetworks.InstanceList.VectorizedInstanceList;
import NeuralNetworks.Nets.RecurrentNeuralNetwork;
import WordToVec.NeuralNetwork;
import WordToVec.WordToVecParameter;
import org.junit.Test;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.LinkedList;
import java.util.Scanner;
import Math.*;

import static org.junit.Assert.*;

public class RecurrentNeuralNetworkTest {

    @Test
    public void testTrain() throws FileNotFoundException, MatrixColumnMismatch, VectorSizeMismatch, MatrixRowColumnMismatch, MatrixDimensionMismatch {
        NeuralNetwork neuralNetwork = new NeuralNetwork(new Corpus("test.txt"), new WordToVecParameter());
        VectorizedDictionary dictionary = neuralNetwork.train();
        VectorizedInstanceList list = new VectorizedInstanceList(new Scanner(new File("data.txt")), dictionary, 1, " ");
        LinkedList<Integer> hiddenLayers = new LinkedList<>();
        hiddenLayers.add(3);
        RecurrentNeuralNetwork net = new RecurrentNeuralNetwork(1, hiddenLayers, list, Activation.LEAKYRELU);
        net.train(1000, 0.01, 0.99, 0.5);
        double accuracy = net.test(list);
        assertEquals(100.0, accuracy, 0.01);
    }
}
