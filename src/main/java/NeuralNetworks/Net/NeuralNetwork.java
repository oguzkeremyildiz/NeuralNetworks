package NeuralNetworks.Net;

import NeuralNetworks.ActivationFunction.*;
import NeuralNetworks.Initializer.Initializer;
import NeuralNetworks.InstanceList.Instance;
import NeuralNetworks.InstanceList.InstanceList;

import java.io.*;
import java.util.LinkedList;
import Math.*;
import NeuralNetworks.Layer.Layer;
import NeuralNetworks.Neuron.Bias;

public class NeuralNetwork extends Net<String> implements Serializable {

    public NeuralNetwork(int seed, LinkedList<Integer> hiddenLayers, InstanceList instanceList, LinkedList<Activation> activation, Initializer initializer) {
        super(seed, activation, instanceList, hiddenLayers);
        int inputSize = instanceList.inputSize();
        if (inputSize > -1) {
            hiddenLayers.set(0, inputSize);
        }
        for (int i = 0; i < hiddenLayers.size(); i++) {
            if (i + 1 < hiddenLayers.size()) {
                this.layers[i] = new Layer(i, hiddenLayers, seed, findInitializerFunction(initializer, hiddenLayers.get(i)));
            } else {
                this.layers[i] = new Layer(hiddenLayers.get(i));
            }
        }
        for (int i = 0; i < biases.length; i++) {
            biases[i] = new Bias(seed, hiddenLayers.get(i + 1), findInitializerFunction(initializer, hiddenLayers.get(i + 1)));
        }
    }

    protected void feedForward() {
        for (int i = 0; i < layers.length - 1; i++) {
            for (int j = 0; j < layers[i + 1].size(); j++) {
                double sum = 0.0;
                for (int k = 0; k < layers[i].size(); k++) {
                    sum += layers[i].getWeight(k, j) * layers[i].getValue(k);
                }
                sum += biases[i].getValue(j);
                if (i + 1 != layers.length - 1) {
                    sum = function.get(i).calculateForward(sum);
                }
                layers[i + 1].setValue(j, sum);
            }
        }
        if (layers[layers.length - 1].size() > 2) {
            layers[layers.length - 1].softmax();
        }
    }

    private void createInputVector(Instance<String> instance) {
        int currentIndex = 0;
        for (int i = 0; i < instance.size() - 1; i++) {
            if (instance.size() - 1 != layers[0].size()) {
                int mapSize = ((InstanceList) instanceList).mapSize(i);
                int featureIndex = ((InstanceList) instanceList).getFeature(i, instance.get(i));
                for (int j = 0; j < mapSize; j++) {
                    if (featureIndex != j) {
                        layers[0].setValue(currentIndex, 0.0);
                    } else {
                        layers[0].setValue(currentIndex, 1.0);
                    }
                    currentIndex++;
                }
            } else {
                layers[0].setValue(i, Double.parseDouble(instance.get(i)));
            }
        }
    }

    protected void setWeights(LinkedList<Matrix> deltaWeights, LinkedList<Matrix> oldDeltaWeights, double momentum) {
        for (int t = 0; t < deltaWeights.size(); t++) {
            Matrix weights = deltaWeights.get(t);
            for (int i = 0; i < weights.getRow(); i++) {
                for (int j = 0; j < weights.getColumn(); j++) {
                    if (!oldDeltaWeights.isEmpty()) {
                        weights.addValue(i, j, momentum * oldDeltaWeights.get(t).getValue(i, j));
                    }
                    if (j > 0) {
                        layers[t].addWeight(j - 1, i, weights.getValue(i, j));
                    } else {
                        biases[t].addWeight(i, weights.getValue(i, j));
                    }
                }
            }
        }
    }

    protected LinkedList<Matrix> backpropagation(int classInfo, double learningRate, double momentum, LinkedList<Matrix> oldDeltaWeights) throws MatrixRowColumnMismatch, MatrixDimensionMismatch {
        LinkedList<Matrix> deltaWeights = new LinkedList<>();
        calculateRMinusY(deltaWeights, classInfo, learningRate);
        for (int i = layers.length - 3; i > -1; i--) {
            calculateError(i, deltaWeights);
            if (i > 0) {
                deltaWeights.addFirst(deltaWeights.getFirst());
                deltaWeights.set(1, deltaWeights.get(1).multiply(layers[i].neuronsToMatrix()));
            } else {
                deltaWeights.set(0, deltaWeights.getFirst().multiply(layers[i].neuronsToMatrix()));
            }
        }
        setWeights(deltaWeights, oldDeltaWeights, momentum);
        return deltaWeights;
    }

    public void train(int epoch, double learningRate, double etaDecrease, double momentum) throws MatrixDimensionMismatch, MatrixRowColumnMismatch {
        LinkedList<Matrix> oldDeltaWeights = new LinkedList<>();
        for (int i = 0; i < epoch; i++) {
            instanceList.shuffle(seed);
            for (int j = 0; j < instanceList.size(); j++) {
                createInputVector(instanceList.getInstance(j));
                feedForward();
                oldDeltaWeights = backpropagation(instanceList.get(instanceList.getInstance(j).getLast()), learningRate, momentum, oldDeltaWeights);
            }
            learningRate *= etaDecrease;
        }
    }

    public String predict(Instance<String> instance) {
        createInputVector(instance);
        feedForward();
        if (instanceList.getOutput() == 1) {
            double outputValue = layers[layers.length - 1].getValue(0);
            if (outputValue >= 0.5) {
                return instanceList.get(1);
            }
            return instanceList.get(0);
        }
        double bestValue = Integer.MIN_VALUE;
        int bestNeuron = -1;
        for (int i = 0; i < layers[layers.length - 1].size(); i++) {
            if (layers[layers.length - 1].getValue(i) > bestValue) {
                bestValue = layers[layers.length - 1].getValue(i);
                bestNeuron = i;
            }
        }
        return instanceList.get(bestNeuron);
    }

    public double test(InstanceList list) {
        int correct = 0;
        int total = 0;
        for (int i = 0; i < list.size(); i++) {
            Instance<String> instance = list.getInstance(i);
            if (instance.getLast().equals(predict(instance))) {
                correct++;
            }
            total++;
        }
        return correct * 100.00 / total;
    }
}