import NeuralNetworks.ActivationFunction.Activation;
import NeuralNetworks.Initializer.Initializer;
import NeuralNetworks.InstanceList.VectorizedInstanceList;
import NeuralNetworks.Net.LSTM;
import Util.FileUtils;
import org.junit.Test;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.LinkedList;
import Math.*;

import static org.junit.Assert.*;

public class LSTMTest {

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
        hiddenLayers.add(20);
        activations.add(Activation.SIGMOID);
        LSTM net = new LSTM(1, hiddenLayers, trainList, activations, Initializer.RANDOM);
        net.train(100, 0.2, 0.99, 0.5);
        if (testList != null) {
            double accuracy = net.test(testList);
            assertEquals(51.68878589986658, accuracy, 0.01);
        }
    }
}
