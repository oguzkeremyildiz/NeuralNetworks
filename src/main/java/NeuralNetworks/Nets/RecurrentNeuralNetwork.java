package NeuralNetworks.Nets;

import NeuralNetworks.ActivationFunctions.Activation;
import NeuralNetworks.Bias;
import NeuralNetworks.Instance;
import NeuralNetworks.InstanceList.VectorizedInstanceList;
import NeuralNetworks.Layer;
import NeuralNetworks.RecurrentLayer;

import java.io.Serializable;
import java.util.*;

public class RecurrentNeuralNetwork extends Net implements Serializable {

    private final Layer[] layers;
    private final VectorizedInstanceList instanceList;
    private final Bias[] biases;

    public RecurrentNeuralNetwork(int seed, LinkedList<Integer> hiddenLayers, VectorizedInstanceList instanceList, Activation activation) {
        super(seed, activation);
        this.instanceList = instanceList;
        hiddenLayers.addFirst(instanceList.getInput());
        hiddenLayers.addLast(instanceList.getOutput());
        this.layers = new Layer[hiddenLayers.size()];
        this.layers[0] = new Layer(hiddenLayers.get(0), hiddenLayers.get(1), seed);
        for (int i = 1; i < hiddenLayers.size(); i++) {
            if (i + 1 < hiddenLayers.size()) {
                this.layers[i] = new RecurrentLayer(hiddenLayers.get(i), hiddenLayers.get(i + 1), seed);
            } else {
                this.layers[i] = new Layer(hiddenLayers.get(i));
            }
        }
        biases = new Bias[hiddenLayers.size() - 1];
        for (int i = 0; i < biases.length; i++) {
            biases[i] = new Bias(seed, layers[i + 1].size());
        }
    }

    private void createInputVector(Vector<String> inputLayer) {
        for (int i = 0; i < layers[0].size(); i++) {
            layers[0].getNeuron(i).setValue(Double.parseDouble(inputLayer.get(i)));
        }
    }

    @Override
    protected void feedForward() {

    }

    @Override
    public void train(int epoch, double learningRate, double etaDecrease, double momentum) {
        for (int i = 0; i < epoch; i++) {
            instanceList.shuffle(seed);
            for (int j = 0; j < instanceList.size(); j++) {
                Instance<Vector<String>> instance = instanceList.getInstance(j);
                for (int k = 0; k < instance.size(); k += 2) {
                    createInputVector(instance.get(k));
                    String classInfo = instance.get(k + 1).get(0);
                    feedForward();
                    
                }
                for (int k = 1; k < layers.length - 1; k++) {
                    ((RecurrentLayer) layers[k]).setValuesToZero();
                }
            }
            learningRate *= etaDecrease;
        }
    }

    public String[] predict(Instance<Vector<String>> instance) {
        return null;
    }

    public double test(VectorizedInstanceList list) {
        return 0.0;
    }
}
