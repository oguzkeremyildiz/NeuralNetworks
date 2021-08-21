package NeuralNetworks.Nets;

import NeuralNetworks.ActivationFunctions.Activation;
import NeuralNetworks.Instance;
import NeuralNetworks.InstanceList.VectorizedInstanceList;

import java.io.Serializable;
import java.util.*;

public class RecurrentNeuralNetwork extends Net<Vector<String>> implements Serializable {

    public RecurrentNeuralNetwork(int seed, LinkedList<Integer> hiddenLayers, VectorizedInstanceList instanceList, Activation activation) {
        super(seed, activation);

    }

    @Override
    protected void feedForward() {

    }

    @Override
    public void train(int epoch, double learningRate, double etaDecrease, double momentum) {

    }

    @Override
    public String predict(Instance<Vector<String>> instance) {
        return null;
    }
}
