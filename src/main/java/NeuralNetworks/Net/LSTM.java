package NeuralNetworks.Net;

import java.io.Serializable;
import java.util.LinkedList;

import Math.*;
import NeuralNetworks.ActivationFunction.Activation;
import NeuralNetworks.ActivationFunction.ActivationFunction;
import NeuralNetworks.ActivationFunction.TanH;
import NeuralNetworks.InstanceList.VectorizedInstanceList;
import NeuralNetworks.Layer.LSTMLayer;
import NeuralNetworks.Neuron.LSTMBias;
import NeuralNetworks.Neuron.LSTMNeuron;

public class LSTM extends RecurrentNeuralNetwork implements Serializable {

    private final ActivationFunction tanh;

    public LSTM(int seed, LinkedList<Integer> hiddenLayers, VectorizedInstanceList instanceList, Activation activation) {
        super(seed, hiddenLayers, instanceList, activation, NetworkType.LSTM);
        tanh = new TanH();
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

    private void calculateForwardVectors(int layer) {
        LSTMLayer lstmLayer = ((LSTMLayer) layers[layer + 1]);
        for (int i = 0; i < layers[layer + 1].size(); i++) {
            lstmLayer.addVector(i, biases[layer].getValue(i));
            lstmLayer.addForgetVector(i, ((LSTMBias) biases[layer]).getForgetGateValue(i));
            lstmLayer.addAddVector(i, ((LSTMBias) biases[layer]).getAddGateValue(i));
            lstmLayer.addGVector(i, ((LSTMBias) biases[layer]).getGGateValue(i));
            for (int j = 0; j < layers[layer].size(); j++) {
                lstmLayer.addVector(i, layers[layer].getNeuron(j).getValue() * layers[layer].getNeuron(j).getWeight(i));
                lstmLayer.addForgetVector(i, layers[layer].getNeuron(j).getValue() * ((LSTMNeuron) layers[layer].getNeuron(j)).getForgetGateWeight(i));
                lstmLayer.addAddVector(i, layers[layer].getNeuron(j).getValue() * ((LSTMNeuron) layers[layer].getNeuron(j)).getAddGateWeight(i));
                lstmLayer.addGVector(i, layers[layer].getNeuron(j).getValue() * ((LSTMNeuron) layers[layer].getNeuron(j)).getGGateWeight(i));
                if (layer + 1 != layers.length - 1) {
                    for (int k = 0; k < layers[layer + 1].size(); k++) {
                        lstmLayer.addVector(i, ((LSTMNeuron) layers[layer + 1].getNeuron(k)).getOldValue() * ((LSTMNeuron) layers[layer + 1].getNeuron(k)).getRecurrentWeight(i));
                        lstmLayer.addForgetVector(i, ((LSTMNeuron) layers[layer + 1].getNeuron(k)).getOldValue() * ((LSTMNeuron) layers[layer + 1].getNeuron(k)).getForgetGateRecurrentWeight(i));
                        lstmLayer.addAddVector(i, ((LSTMNeuron) layers[layer + 1].getNeuron(k)).getOldValue() * ((LSTMNeuron) layers[layer + 1].getNeuron(k)).getAddGateRecurrentWeight(i));
                        lstmLayer.addGVector(i, ((LSTMNeuron) layers[layer + 1].getNeuron(k)).getOldValue() * ((LSTMNeuron) layers[layer + 1].getNeuron(k)).getGGateRecurrentWeight(i));
                    }
                }
            }
        }
        for (int i = 0; i < layers[layer + 1].size(); i++) {
            lstmLayer.setVector(i, function.calculateForward(lstmLayer.getVector(i)));
            lstmLayer.setForgetVector(i, function.calculateForward(lstmLayer.getForgetVector(i)));
            lstmLayer.setAddVector(i, function.calculateForward(lstmLayer.getAddVector(i)));
        }
        for (int i = 0; i < layers[layer + 1].size(); i++) {
            double value = lstmLayer.getGVector(i);
            lstmLayer.setGVector(i, tanh.calculateForward(value));
        }
    }

    @Override
    protected void feedForward() throws VectorSizeMismatch {
        for (int i = 0; i < layers.length - 2; i++) {
            LSTMLayer layer = (LSTMLayer) layers[i + 1];
            layer.clear();
            calculateForwardVectors(i);
            Vector k = layer.forgetVectorElementProduct(((LSTMLayer) layers[i + 1]).oldContextValuesToVector());
            Vector j = layer.addVectorElementProduct();
            for (int l = 0; l < layers[i + 1].size(); l++) {
                ((LSTMNeuron) layers[i + 1].getNeuron(l)).setContextValue(j.getValue(l) + k.getValue(l));
                layers[i + 1].getNeuron(l).setValue(layer.getVector(l) * tanh.calculateForward(((LSTMNeuron) layers[i + 1].getNeuron(l)).getContextValue()));
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

    protected void calculateRMinusY(LinkedList<Matrix> deltaWeights, int classInfo, double learningRate) throws MatrixRowColumnMismatch {
        super.calculateRMinusY(deltaWeights, classInfo, learningRate);
        Matrix rMinusY = deltaWeights.getFirst();
        for (int i = 0; i < 3; i++) {
            deltaWeights.addFirst(rMinusY);
        }
    }

    private LinkedList<Matrix> calculateErrors(int i, LinkedList<Matrix> deltaWeights) throws MatrixDimensionMismatch, MatrixRowColumnMismatch {
        LinkedList<Matrix> errors = new LinkedList<>();

        return errors;
    }

    @Override
    protected LinkedList<Matrix> backpropagation(int classInfo, double learningRate, double momentum, LinkedList<Matrix> oldDeltaWeights) throws MatrixRowColumnMismatch, MatrixDimensionMismatch {
        LinkedList<Matrix> deltaWeights = new LinkedList<>();
        calculateRMinusY(deltaWeights, classInfo, learningRate);
        for (int i = layers.length - 3; i > -1; i--) {
            LinkedList<Matrix> errors = calculateErrors(i, deltaWeights);
            deltaWeights.set(0, deltaWeights.getFirst().multiply(layers[i].neuronsToMatrix()));
            deltaWeights.set(1, deltaWeights.get(1).multiply(layers[i].neuronsToMatrix()));
            deltaWeights.set(2, deltaWeights.get(2).multiply(layers[i].neuronsToMatrix()));
            deltaWeights.set(3, deltaWeights.get(3).multiply(layers[i].neuronsToMatrix()));
            deltaWeights.add(4, errors.getFirst().multiply(((LSTMLayer) layers[i + 1]).oldNeuronsToMatrix()));
            deltaWeights.add(5, errors.get(1).multiply(((LSTMLayer) layers[i + 1]).oldNeuronsToMatrix()));
            deltaWeights.add(6, errors.get(2).multiply(((LSTMLayer) layers[i + 1]).oldNeuronsToMatrix()));
            deltaWeights.add(7, errors.get(3).multiply(((LSTMLayer) layers[i + 1]).oldNeuronsToMatrix()));
            if (i > 0) {
                deltaWeights.addFirst(errors.getFirst());
                deltaWeights.add(1, errors.get(1));
                deltaWeights.add(2, errors.get(2));
                deltaWeights.add(3, errors.get(3));
            }
        }
        setWeights(deltaWeights, oldDeltaWeights, momentum);
        return deltaWeights;
    }
}
