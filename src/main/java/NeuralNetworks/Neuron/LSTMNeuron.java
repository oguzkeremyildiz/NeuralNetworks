package NeuralNetworks.Neuron;

import java.io.Serializable;

public class LSTMNeuron extends RecurrentNeuron implements Serializable {

    private final double[] forgetGate;
    private final double[] forgetGateRecurrent;
    private final double[] addGate;
    private final double[] addGateRecurrent;
    private final double[] gGate;
    private final double[] gGateRecurrent;
    private double oldContextValue, contextValue;

    public LSTMNeuron(double[] weights, double[] recurrentWeights, double[] forgetGate, double[] forgetGateRecurrent, double[] addGate, double[] addGateRecurrent, double[] gGate, double[] gGateRecurrent) {
        super(weights, recurrentWeights);
        this.forgetGate = forgetGate;
        this.forgetGateRecurrent = forgetGateRecurrent;
        this.addGate = addGate;
        this.addGateRecurrent = addGateRecurrent;
        this.gGate = gGate;
        this.gGateRecurrent = gGateRecurrent;
        this.contextValue = 0.0;
        this.oldContextValue = 0.0;
    }

    public LSTMNeuron(double[] weights, double[] forgetGate, double[] addGate, double[] gGate) {
        super(weights);
        this.forgetGate = forgetGate;
        this.forgetGateRecurrent = null;
        this.addGate = addGate;
        this.addGateRecurrent = null;
        this.gGate = gGate;
        this.gGateRecurrent = null;
        this.contextValue = 0.0;
        this.oldContextValue = 0.0;
    }

    public double getForgetGateWeight(int i) {
        return forgetGate[i];
    }

    public void addForgetGateWeight(int index, double value) {
        forgetGate[index] += value;
    }

    public void addAddGateWeight(int index, double value) {
        addGate[index] += value;
    }

    public void addGGateWeight(int index, double value) {
        gGate[index] += value;
    }

    public double getForgetGateRecurrentWeight(int i) {
        return forgetGateRecurrent[i];
    }

    public void addForgetGateRecurrentWeight(int index, double value) {
        forgetGateRecurrent[index] += value;
    }

    public double getAddGateWeight(int i) {
        return addGate[i];
    }

    public double getAddGateRecurrentWeight(int i) {
        return addGateRecurrent[i];
    }

    public void addAddGateRecurrentWeight(int index, double value) {
        addGateRecurrent[index] += value;
    }

    public double getGGateWeight(int i) {
        return gGate[i];
    }

    public double getGGateRecurrentWeight(int i) {
        return gGateRecurrent[i];
    }

    public void addGGateRecurrentWeight(int index, double value) {
        gGateRecurrent[index] += value;
    }

    public double getContextValue() {
        return contextValue;
    }

    public void setContextValue(double contextValue) {
        this.contextValue = contextValue;
    }

    public double getOldContextValue() {
        return oldContextValue;
    }

    public void setOldContextValue(double oldContextValue) {
        this.oldContextValue = oldContextValue;
    }

    public void setOldContextValue() {
        this.oldContextValue = this.contextValue;
    }
}
