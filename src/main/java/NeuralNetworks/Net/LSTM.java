package NeuralNetworks.Net;

import java.io.Serializable;
import java.util.LinkedList;

import Math.*;
import NeuralNetworks.ActivationFunction.Activation;
import NeuralNetworks.ActivationFunction.ActivationFunction;
import NeuralNetworks.ActivationFunction.TanH;
import NeuralNetworks.InstanceList.VectorizedInstanceList;
import NeuralNetworks.Layer.LSTMLayer;
import NeuralNetworks.Neuron.LSTMNeuron;

public class LSTM extends RecurrentNeuralNetwork implements Serializable {

    private final ActivationFunction tanh;

    public LSTM(int seed, LinkedList<Integer> hiddenLayers, VectorizedInstanceList instanceList, Activation activation) {
        super(seed, hiddenLayers, instanceList, activation, NetworkType.LSTM);
        if (activation.equals(Activation.TANH)) {
            tanh = null;
        } else {
            tanh = new TanH();
        }
    }

    protected void setOldValues() {
        for (int k = 1; k < layers.length - 1; k++) {
            ((LSTMLayer) layers[k]).setValues();
            ((LSTMLayer) layers[k]).setContextValues();
        }
    }

    protected void setOldValuesToZero() {
        for (int k = 1; k < layers.length - 1; k++) {
            ((LSTMLayer) layers[k]).setValuesToZero();
            ((LSTMLayer) layers[k]).setContextValuesToZero();
        }
    }

    private LinkedList<Vector> calculateForwardVectors(int layer) {
        LinkedList<Vector> vectors = new LinkedList<>();
        for (int i = 0; i < 4; i++) {
            vectors.add(new Vector(layers[layer + 1].size(), 0));
        }
        for (int i = 0; i < layers[layer + 1].size(); i++) {
            vectors.get(0).addValue(i, biases[layer].getValue(i));
            for (int j = 0; j < layers[layer].size(); j++) {
                vectors.get(0).addValue(i, layers[layer].getNeuron(j).getValue() * layers[layer].getNeuron(j).getWeight(i));
                vectors.get(1).addValue(i, layers[layer].getNeuron(j).getValue() * ((LSTMNeuron) layers[layer].getNeuron(j)).getForgetGateWeight(i));
                vectors.get(2).addValue(i, layers[layer].getNeuron(j).getValue() * ((LSTMNeuron) layers[layer].getNeuron(j)).getAddGateWeight(i));
                vectors.get(3).addValue(i, layers[layer].getNeuron(j).getValue() * ((LSTMNeuron) layers[layer].getNeuron(j)).getGGateWeight(i));
                if (layer + 1 != layers.length - 1) {
                    for (int k = 0; k < layers[layer + 1].size(); k++) {
                        vectors.get(0).addValue(i, ((LSTMNeuron) layers[layer + 1].getNeuron(k)).getOldValue() * ((LSTMNeuron) layers[layer + 1].getNeuron(k)).getRecurrentWeight(i));
                        vectors.get(1).addValue(i, ((LSTMNeuron) layers[layer + 1].getNeuron(k)).getOldValue() * ((LSTMNeuron) layers[layer + 1].getNeuron(k)).getForgetGateRecurrentWeight(i));
                        vectors.get(2).addValue(i, ((LSTMNeuron) layers[layer + 1].getNeuron(k)).getOldValue() * ((LSTMNeuron) layers[layer + 1].getNeuron(k)).getAddGateRecurrentWeight(i));
                        vectors.get(3).addValue(i, ((LSTMNeuron) layers[layer + 1].getNeuron(k)).getOldValue() * ((LSTMNeuron) layers[layer + 1].getNeuron(k)).getGGateRecurrentWeight(i));
                    }
                }
            }
        }
        for (int i = 0; i < vectors.size() - 1; i++) {
            for (int j = 0; j < vectors.get(i).size(); j++) {
                vectors.get(i).setValue(j, function.calculateForward(vectors.get(i).getValue(j)));
            }
        }
        for (int i = 0; i < vectors.getLast().size(); i++) {
            double value = vectors.getLast().getValue(i);
            if (tanh == null) {
                vectors.getLast().setValue(i, function.calculateForward(value));
            } else {
                vectors.getLast().setValue(i, tanh.calculateForward(value));
            }
        }
        return vectors;
    }

    @Override
    protected void feedForward() throws VectorSizeMismatch {
        for (int i = 0; i < layers.length - 2; i++) {
            LinkedList<Vector> vectors = calculateForwardVectors(i);
            Vector k = vectors.get(1).elementProduct(((LSTMLayer) layers[i + 1]).oldContextValuesToVector());
            Vector j = vectors.get(2).elementProduct(vectors.get(3));
            for (int l = 0; l < layers[i + 1].size(); l++) {
                ((LSTMNeuron) layers[i + 1].getNeuron(l)).setContextValue(j.getValue(l) + k.getValue(l));
                if (tanh == null) {
                    layers[i + 1].getNeuron(l).setValue(vectors.getFirst().getValue(l) * function.calculateForward(((LSTMNeuron) layers[i + 1].getNeuron(l)).getContextValue()));
                } else {
                    layers[i + 1].getNeuron(l).setValue(vectors.getFirst().getValue(l) * tanh.calculateForward(((LSTMNeuron) layers[i + 1].getNeuron(l)).getContextValue()));
                }
            }
        }
        for (int i = 0; i < layers[layers.length - 1].size(); i++) {
            double sum = 0.0;
            for (int j = 0; j < layers[layers.length - 2].size(); j++) {
                sum += layers[layers.length - 2].getNeuron(j).getValue() * layers[layers.length - 2].getNeuron(j).getWeight(i);
            }
            layers[layers.length - 1].getNeuron(i).setValue(sum);
        }
        if (layers[layers.length - 1].size() > 2) {
            layers[layers.length - 1].softmax();
        }
    }

    @Override
    protected void setWeights(LinkedList<Matrix> deltaWeights, LinkedList<Matrix> oldDeltaWeights, double momentum) {

    }

    @Override
    protected LinkedList<Matrix> backpropagation(int classInfo, double learningRate, double momentum, LinkedList<Matrix> oldDeltaWeights) {
        return null;
    }
}
