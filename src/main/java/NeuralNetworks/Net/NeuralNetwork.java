package NeuralNetworks.Net;

import NeuralNetworks.ActivationFunction.*;
import NeuralNetworks.InstanceList.Instance;
import NeuralNetworks.InstanceList.InstanceList;

import java.io.*;
import java.util.LinkedList;
import Math.*;

public class NeuralNetwork extends Net<String> implements Serializable {

    public NeuralNetwork(int seed, LinkedList<Integer> hiddenLayers, InstanceList instanceList, Activation activation) {
        super(seed, activation, instanceList, hiddenLayers, NetworkType.NEURALNETWORK);
    }

    protected void feedForward() {
        for (int i = 0; i < layers.length - 1; i++) {
            for (int j = 0; j < layers[i + 1].size(); j++) {
                double sum = 0.0;
                for (int k = 0; k < layers[i].size(); k++) {
                    sum += layers[i].getNeuron(k).getWeight(j) * layers[i].getNeuron(k).getValue();
                }
                sum += biases[i].getValue(j);
                if (i + 1 != layers.length - 1) {
                    sum = function.calculateForward(sum);
                }
                layers[i + 1].getNeuron(j).setValue(sum);
            }
        }
        if (layers[layers.length - 1].size() > 2) {
            layers[layers.length - 1].softmax();
        }
    }

    private void createInputVector(Instance<String> instance) {
        for (int i = 0; i < layers[0].size(); i++) {
            layers[0].getNeuron(i).setValue(Double.parseDouble(instance.get(i)));
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
                        layers[t].getNeuron(j - 1).addWeight(i, weights.getValue(i, j));
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
            Matrix currentError = calculateError(i, deltaWeights);
            deltaWeights.set(0, deltaWeights.getFirst().multiply(layers[i].neuronsToMatrix()));
            if (i > 0) {
                deltaWeights.addFirst(currentError);
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
            double outputValue = layers[layers.length - 1].getNeuron(0).getValue();
            if (outputValue >= 0.5) {
                return instanceList.get(1);
            }
            return instanceList.get(0);
        }
        double bestValue = Integer.MIN_VALUE;
        int bestNeuron = -1;
        for (int i = 0; i < layers[layers.length - 1].size(); i++) {
            if (layers[layers.length - 1].getNeuron(i).getValue() > bestValue) {
                bestValue = layers[layers.length - 1].getNeuron(i).getValue();
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