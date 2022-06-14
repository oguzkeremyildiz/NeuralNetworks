package NeuralNetworks.Net;

import java.io.Serializable;
import java.util.LinkedList;

import Math.*;
import NeuralNetworks.ActivationFunction.Activation;
import NeuralNetworks.ActivationFunction.ActivationFunction;
import NeuralNetworks.ActivationFunction.TanH;
import NeuralNetworks.Initializer.Initializer;
import NeuralNetworks.InstanceList.VectorizedInstanceList;
import NeuralNetworks.Layer.LSTMLayer;
import NeuralNetworks.Layer.Layer;
import NeuralNetworks.Neuron.Bias;
import NeuralNetworks.Neuron.LSTMBias;

public class LSTM extends RecurrentNeuralNetwork implements Serializable {

    private final ActivationFunction tanh;

    public LSTM(int seed, LinkedList<Integer> hiddenLayers, VectorizedInstanceList instanceList, LinkedList<Activation> activation, Initializer initializer) {
        super(seed, hiddenLayers, instanceList, activation, initializer);
        tanh = new TanH();
        this.layers[0] = new LSTMLayer(0, hiddenLayers, seed, false, findInitializerFunction(initializer, hiddenLayers.get(0)));
        for (int i = 1; i < hiddenLayers.size(); i++) {
            if (i + 1 < hiddenLayers.size()) {
                this.layers[i] = new LSTMLayer(i, hiddenLayers, seed, true, findInitializerFunction(initializer, hiddenLayers.get(i)));
            } else {
                this.layers[i] = new Layer(hiddenLayers.get(i));
            }
        }
        for (int i = 0; i < biases.length - 1; i++) {
            biases[i] = new LSTMBias(seed, hiddenLayers.get(i + 1), findInitializerFunction(initializer, hiddenLayers.get(i + 1)));
        }
        biases[biases.length - 1] = new Bias(seed, hiddenLayers.get(biases.length), findInitializerFunction(initializer, hiddenLayers.get(biases.length)));
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
        for (int i = 0; i < lstmLayer.size(); i++) {
            lstmLayer.getVector(0).addValue(i, biases[layer].getValue(i));
            lstmLayer.getVector(1).addValue(i, ((LSTMBias) biases[layer]).getGateValue(0, i));
            lstmLayer.getVector(2).addValue(i, ((LSTMBias) biases[layer]).getGateValue(1, i));
            lstmLayer.getVector(3).addValue(i, ((LSTMBias) biases[layer]).getGateValue(2, i));
            for (int j = 0; j < layers[layer].size(); j++) {
                lstmLayer.getVector(0).addValue(i, layers[layer].getValue(j) * layers[layer].getWeight(j, i));
                lstmLayer.getVector(1).addValue(i, layers[layer].getValue(j) * ((LSTMLayer) layers[layer]).getGateWeight(0, j, i));
                lstmLayer.getVector(2).addValue(i, layers[layer].getValue(j) * ((LSTMLayer) layers[layer]).getGateWeight(1, j, i));
                lstmLayer.getVector(3).addValue(i, layers[layer].getValue(j) * ((LSTMLayer) layers[layer]).getGateWeight(2, j, i));
                if (layer + 1 != layers.length - 1) {
                    for (int k = 0; k < layers[layer + 1].size(); k++) {
                        lstmLayer.getVector(0).addValue(i, lstmLayer.getOldValue(k) * lstmLayer.getRecurrentWeight(k, i));
                        lstmLayer.getVector(1).addValue(i, lstmLayer.getOldValue(k) * lstmLayer.getGateRecurrentWeight(0, k, i));
                        lstmLayer.getVector(2).addValue(i, lstmLayer.getOldValue(k) * lstmLayer.getGateRecurrentWeight(1, k, i));
                        lstmLayer.getVector(3).addValue(i, lstmLayer.getOldValue(k) * lstmLayer.getGateRecurrentWeight(2, k, i));
                    }
                }
            }
        }
        for (int i = 0; i < layers[layer + 1].size(); i++) {
            lstmLayer.getVector(0).setValue(i, function.get(layer).calculateForward(lstmLayer.getVector(0).getValue(i)));
            lstmLayer.getVector(1).setValue(i, function.get(layer).calculateForward(lstmLayer.getVector(1).getValue(i)));
            lstmLayer.getVector(2).setValue(i, function.get(layer).calculateForward(lstmLayer.getVector(2).getValue(i)));
        }
        for (int i = 0; i < layers[layer + 1].size(); i++) {
            double value = lstmLayer.getVector(3).getValue(i);
            lstmLayer.getVector(3).setValue(i, tanh.calculateForward(value));
        }
    }

    @Override
    protected void feedForward() throws VectorSizeMismatch {
        for (int i = 0; i < layers.length - 2; i++) {
            LSTMLayer layer = (LSTMLayer) layers[i + 1];
            layer.clear();
            calculateForwardVectors(i);
            Vector k = layer.getVector(1).elementProduct(layer.oldContextValuesToVector());
            Vector j = layer.getVector(2).elementProduct(layer.getVector(3));
            for (int l = 0; l < layer.size(); l++) {
                layer.setContextValue(l, j.getValue(l) + k.getValue(l));
                double value = tanh.calculateForward(layer.getContextValue(l));
                layer.getVector(4).setValue(l, value);
                layer.setValue(l, layer.getVector(0).getValue(l) * value);
            }
        }
        for (int i = 0; i < layers[layers.length - 1].size(); i++) {
            double sum = 0.0;
            for (int j = 0; j < layers[layers.length - 2].size(); j++) {
                sum += layers[layers.length - 2].getValue(j) * layers[layers.length - 2].getWeight(j, i);
            }
            layers[layers.length - 1].setValue(i, sum);
        }
        if (layers[layers.length - 1].size() > 2) {
            layers[layers.length - 1].softmax();
        }
    }

    @Override
    protected void setWeights(LinkedList<Matrix> deltaWeights, LinkedList<Matrix> oldDeltaWeights, double momentum) {
        for (int t = 0; t < deltaWeights.size() - 1; t += 8) {
            Matrix weights = deltaWeights.get(t);
            Matrix forgetWeights = deltaWeights.get(t + 1);
            Matrix addWeights = deltaWeights.get(t + 2);
            Matrix gWeights = deltaWeights.get(t + 3);
            for (int i = 0; i < weights.getRow(); i++) {
                for (int j = 0; j < weights.getColumn(); j++) {
                    if (!oldDeltaWeights.isEmpty()) {
                        weights.addValue(i, j, momentum * oldDeltaWeights.get(t).getValue(i, j));
                        forgetWeights.addValue(i, j, momentum * oldDeltaWeights.get(t + 1).getValue(i, j));
                        addWeights.addValue(i, j, momentum * oldDeltaWeights.get(t + 2).getValue(i, j));
                        gWeights.addValue(i, j, momentum * oldDeltaWeights.get(t + 3).getValue(i, j));
                    }
                    if (j > 0) {
                        layers[t / 8].addWeight(j - 1, i, weights.getValue(i, j));
                        ((LSTMLayer) layers[t / 8]).addGateWeight(0, j - 1, i, forgetWeights.getValue(i, j));
                        ((LSTMLayer) layers[t / 8]).addGateWeight(1, j - 1, i, addWeights.getValue(i, j));
                        ((LSTMLayer) layers[t / 8]).addGateWeight(2, j - 1, i, gWeights.getValue(i, j));
                    } else {
                        biases[t / 8].addWeight(i, weights.getValue(i, j));
                        ((LSTMBias) biases[t / 8]).addGateValue(0, i, forgetWeights.getValue(i, j));
                        ((LSTMBias) biases[t / 8]).addGateValue(1, i, addWeights.getValue(i, j));
                        ((LSTMBias) biases[t / 8]).addGateValue(2, i, gWeights.getValue(i, j));
                    }
                }
            }
            Matrix recurrentWeights = deltaWeights.get(t + 4);
            Matrix recurrentForgetWeights = deltaWeights.get(t + 5);
            Matrix recurrentAddWeights = deltaWeights.get(t + 6);
            Matrix recurrentGWeights = deltaWeights.get(t + 7);
            for (int i = 0; i < recurrentWeights.getRow(); i++) {
                for (int j = 0; j < recurrentWeights.getColumn(); j++) {
                    if (!oldDeltaWeights.isEmpty()) {
                        recurrentWeights.addValue(i, j, momentum * oldDeltaWeights.get(t + 4).getValue(i, j));
                        recurrentForgetWeights.addValue(i, j, momentum * oldDeltaWeights.get(t + 5).getValue(i, j));
                        recurrentAddWeights.addValue(i, j, momentum * oldDeltaWeights.get(t + 6).getValue(i, j));
                        recurrentGWeights.addValue(i, j, momentum * oldDeltaWeights.get(t + 7).getValue(i, j));
                    }
                    ((LSTMLayer) layers[(t / 8) + 1]).addRecurrentWeight(j, i, recurrentWeights.getValue(i, j));
                    ((LSTMLayer) layers[(t / 8) + 1]).addGateRecurrentWeight(0, j, i, recurrentForgetWeights.getValue(i, j));
                    ((LSTMLayer) layers[(t / 8) + 1]).addGateRecurrentWeight(1, j, i, recurrentAddWeights.getValue(i, j));
                    ((LSTMLayer) layers[(t / 8) + 1]).addGateRecurrentWeight(2, j, i, recurrentGWeights.getValue(i, j));
                }
            }
        }
        setLastWeights(deltaWeights.getLast(), oldDeltaWeights, momentum);
    }

    protected void calculateRMinusY(LinkedList<Matrix> deltaWeights, int classInfo, double learningRate) throws MatrixRowColumnMismatch {
        super.calculateRMinusY(deltaWeights, classInfo, learningRate);
        Matrix rMinusY = deltaWeights.getFirst();
        for (int i = 0; i < 3; i++) {
            deltaWeights.addFirst(rMinusY);
        }
    }

    protected void calculateError(int i, LinkedList<Matrix> deltaWeights) throws MatrixRowColumnMismatch, MatrixDimensionMismatch {
        deltaWeights.set(0, layers[i + 1].weightsToMatrix().multiply(deltaWeights.getFirst()));
        if (i == layers.length - 3) {
            deltaWeights.set(1, layers[i + 1].weightsToMatrix().multiply(deltaWeights.get(1)));
            deltaWeights.set(2, layers[i + 1].weightsToMatrix().multiply(deltaWeights.get(2)));
            deltaWeights.set(3, layers[i + 1].weightsToMatrix().multiply(deltaWeights.get(3)));
        } else {
            deltaWeights.set(1, ((LSTMLayer) layers[i + 1]).forgetGateWeightsToMatrix().multiply(deltaWeights.get(1)));
            deltaWeights.set(2, ((LSTMLayer) layers[i + 1]).addGateWeightsToMatrix().multiply(deltaWeights.get(2)));
            deltaWeights.set(3, ((LSTMLayer) layers[i + 1]).gGateWeightsToMatrix().multiply(deltaWeights.get(3)));
        }
        deltaWeights.set(0, deltaWeights.getFirst().elementProduct(((LSTMLayer) layers[i + 1]).getVector(4)).elementProduct(function.get(i).calculateBack(((LSTMLayer) layers[i + 1]).getVector(0))));
        deltaWeights.set(1, deltaWeights.get(1).elementProduct(((LSTMLayer) layers[i + 1]).getVector(0)).elementProduct(tanh.calculateBack(((LSTMLayer) layers[i + 1]).getVector(4))).elementProduct(((LSTMLayer) layers[i + 1]).oldContextValuesToVector()).elementProduct(function.get(i).calculateBack(((LSTMLayer) layers[i + 1]).getVector(1))));
        deltaWeights.set(2, deltaWeights.get(2).elementProduct(((LSTMLayer) layers[i + 1]).getVector(0)).elementProduct(tanh.calculateBack(((LSTMLayer) layers[i + 1]).getVector(4))).elementProduct(function.get(i).calculateBack(((LSTMLayer) layers[i + 1]).getVector(2))));
        deltaWeights.set(3, deltaWeights.get(3).elementProduct(((LSTMLayer) layers[i + 1]).getVector(0)).elementProduct(tanh.calculateBack(((LSTMLayer) layers[i + 1]).getVector(4))).elementProduct(((LSTMLayer) layers[i + 1]).getVector(2)).elementProduct(tanh.calculateBack(((LSTMLayer) layers[i + 1]).getVector(3))));
    }

    @Override
    protected LinkedList<Matrix> backpropagation(int classInfo, double learningRate, double momentum, LinkedList<Matrix> oldDeltaWeights) throws MatrixRowColumnMismatch, MatrixDimensionMismatch {
        LinkedList<Matrix> deltaWeights = new LinkedList<>();
        calculateRMinusY(deltaWeights, classInfo, learningRate);
        for (int i = layers.length - 3; i > -1; i--) {
            calculateError(i, deltaWeights);
            if (i > 0) {
                deltaWeights.addFirst(deltaWeights.getFirst());
                deltaWeights.add(1, deltaWeights.get(1));
                deltaWeights.add(2, deltaWeights.get(2));
                deltaWeights.add(3, deltaWeights.get(3));
                deltaWeights.add(8, deltaWeights.get(4).multiply(((LSTMLayer) layers[i + 1]).oldNeuronsToMatrix()));
                deltaWeights.add(9, deltaWeights.get(5).multiply(((LSTMLayer) layers[i + 1]).oldNeuronsToMatrix()));
                deltaWeights.add(10, deltaWeights.get(6).multiply(((LSTMLayer) layers[i + 1]).oldNeuronsToMatrix()));
                deltaWeights.add(11, deltaWeights.get(7).multiply(((LSTMLayer) layers[i + 1]).oldNeuronsToMatrix()));
                deltaWeights.set(4, deltaWeights.get(4).multiply(layers[i].neuronsToMatrix()));
                deltaWeights.set(5, deltaWeights.get(5).multiply(layers[i].neuronsToMatrix()));
                deltaWeights.set(6, deltaWeights.get(6).multiply(layers[i].neuronsToMatrix()));
                deltaWeights.set(7, deltaWeights.get(7).multiply(layers[i].neuronsToMatrix()));
            } else {
                deltaWeights.add(4, deltaWeights.getFirst().multiply(((LSTMLayer) layers[i + 1]).oldNeuronsToMatrix()));
                deltaWeights.add(5, deltaWeights.get(1).multiply(((LSTMLayer) layers[i + 1]).oldNeuronsToMatrix()));
                deltaWeights.add(6, deltaWeights.get(2).multiply(((LSTMLayer) layers[i + 1]).oldNeuronsToMatrix()));
                deltaWeights.add(7, deltaWeights.get(3).multiply(((LSTMLayer) layers[i + 1]).oldNeuronsToMatrix()));
                deltaWeights.set(0, deltaWeights.getFirst().multiply(layers[i].neuronsToMatrix()));
                deltaWeights.set(1, deltaWeights.get(1).multiply(layers[i].neuronsToMatrix()));
                deltaWeights.set(2, deltaWeights.get(2).multiply(layers[i].neuronsToMatrix()));
                deltaWeights.set(3, deltaWeights.get(3).multiply(layers[i].neuronsToMatrix()));
            }
        }
        setWeights(deltaWeights, oldDeltaWeights, momentum);
        return deltaWeights;
    }
}
