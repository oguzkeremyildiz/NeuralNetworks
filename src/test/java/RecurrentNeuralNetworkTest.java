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
    public void testTrain() throws FileNotFoundException, MatrixColumnMismatch, VectorSizeMismatch {
        Corpus corpus = new Corpus("test.txt");
        WordToVecParameter parameter = new WordToVecParameter();
        parameter.setCbow(true);
        NeuralNetwork neuralNetwork = new NeuralNetwork(corpus, parameter);
        VectorizedDictionary dic = neuralNetwork.train();
        VectorizedInstanceList list = new VectorizedInstanceList(new Scanner(new File("data.txt")), dic, 1, " ");
        LinkedList<Integer> hiddenLayers = new LinkedList<>();
        hiddenLayers.add(20);
        RecurrentNeuralNetwork net = new RecurrentNeuralNetwork(1, hiddenLayers, list, Activation.SIGMOID);
        net.train(1000, 0.01, 0.99, 0.5);
        double accuracy = net.test(list);
        assertEquals(0.0, accuracy, 0.01);
    }
}
