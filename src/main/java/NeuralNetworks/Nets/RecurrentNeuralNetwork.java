package NeuralNetworks.Nets;

import NeuralNetworks.*;
import NeuralNetworks.ActivationFunctions.Activation;
import NeuralNetworks.InstanceList.VectorizedInstanceList;

import java.io.Serializable;
import java.util.*;
import Math.*;

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

    private void createInputVector(java.util.Vector<String> inputLayer) {
        for (int i = 0; i < layers[0].size(); i++) {
            layers[0].getNeuron(i).setValue(Double.parseDouble(inputLayer.get(i)));
        }
    }

    @Override
    protected void feedForward() {
        for (int i = 0; i < layers.length - 1; i++) {
            for (int j = 0; j < layers[i + 1].size(); j++) {
                double sum = 0.0;
                for (int k = 0; k < layers[i].size(); k++) {
                    sum += layers[i].getNeuron(k).getWeight(j) * layers[i].getNeuron(k).getValue();
                }
                sum += biases[i].getValue(j);
                if (i + 1 != layers.length - 1) {
                    for (int k = 0; k < layers[i + 1].size(); k++) {
                        sum += ((RecurrentNeuron) layers[i + 1].getNeuron(k)).getRecurrentWeight(j) * ((RecurrentNeuron) layers[i + 1].getNeuron(k)).getOldValue();
                    }
                    sum = function.calculateForward(sum);
                }
                layers[i + 1].getNeuron(j).setValue(sum);
            }
        }
        if (layers[layers.length - 1].size() > 2) {
            layers[layers.length - 1].softmax();
        }
        for (int k = 1; k < layers.length - 1; k++) {
            ((RecurrentLayer) layers[k]).setValues();
        }
    }

    @Override
    protected LinkedList<Matrix> backpropagation(int classInfo, double learningRate, double momentum, LinkedList<Matrix> oldDeltaWeights) {
        return null;
    }

    @Override
    public void train(int epoch, double learningRate, double etaDecrease, double momentum) {
        LinkedList<Matrix> oldDeltaWeights = new LinkedList<>();
        for (int i = 0; i < epoch; i++) {
            instanceList.shuffle(seed);
            for (int j = 0; j < instanceList.size(); j++) {
                Instance<java.util.Vector<String>> instance = instanceList.getInstance(j);
                for (int k = 0; k < instance.size(); k += 2) {
                    createInputVector(instance.get(k));
                    String classInfo = instance.get(k + 1).get(0);
                    feedForward();
                    oldDeltaWeights = backpropagation(instanceList.get(classInfo), learningRate, momentum, oldDeltaWeights);
                }
                for (int k = 1; k < layers.length - 1; k++) {
                    ((RecurrentLayer) layers[k]).setValuesToZero();
                }
            }
            learningRate *= etaDecrease;
        }
    }

    public LinkedList<String> predict(Instance<java.util.Vector<String>> instance) {
        for (int k = 1; k < layers.length - 1; k++) {
            ((RecurrentLayer) layers[k]).setValuesToZero();
        }
        LinkedList<String> classes = new LinkedList<>();
        for (int i = 0; i < instance.size(); i += 2) {
            createInputVector(instance.get(i));
            feedForward();
            if (instanceList.getOutput() == 1) {
                double outputValue = layers[layers.length - 1].getNeuron(0).getValue();
                if (outputValue >= 0.5) {
                    classes.add(instanceList.get(1));
                }
                classes.add(instanceList.get(0));
            } else {
                double bestValue = Integer.MIN_VALUE;
                int bestNeuron = -1;
                for (int j = 0; j < layers[layers.length - 1].size(); j++) {
                    if (layers[layers.length - 1].getNeuron(j).getValue() > bestValue) {
                        bestValue = layers[layers.length - 1].getNeuron(j).getValue();
                        bestNeuron = j;
                    }
                }
                classes.add(instanceList.get(bestNeuron));
            }
        }
        return classes;
    }

    public double test(VectorizedInstanceList list) {
        int correct = 0;
        int total = 0;
        for (int i = 0; i < list.size(); i++) {
            Instance<java.util.Vector<String>> instance = list.getInstance(i);
            LinkedList<String> predictedList = predict(instance);
            LinkedList<String> actualList = list.collectClassInfos(instance);
            for (int j = 0; j < actualList.size(); j++) {
                if (actualList.get(j).equals(predictedList.get(j))) {
                    correct++;
                }
                total++;
            }
        }
        return correct * 100.00 / total;
    }
}
