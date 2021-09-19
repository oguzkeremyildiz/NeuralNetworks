import NeuralNetworks.ActivationFunction.Activation;
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
        hiddenLayers.add(15);
        LSTM net = new LSTM(1, hiddenLayers, trainList, Activation.RELU);
        net.train(100, 0.2, 0.99, 0.0);
        if (testList != null) {
            double accuracy = net.test(testList);
            assertEquals(17.702408538726214, accuracy, 0.01);
        }
    }
}
