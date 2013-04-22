package org.wicht.api;

public class Synapse {
    private final Neuron inputNeuron;
    private final Neuron outputNeuron;

    private double weight;
    private double prevDeltaWeight;
    private double deltaWeight;

    public Synapse(Neuron inputNeuron, Neuron outputNeuron) {
        super();

        this.inputNeuron = inputNeuron;
        this.outputNeuron = outputNeuron;

        weight = Math.random() * 2.0 - 1.0;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

    public Neuron getInputNeuron() {
        return inputNeuron;
    }

    public Neuron getOutputNeuron() {
        return outputNeuron;
    }

    public double getDeltaWeight() {
        return deltaWeight;
    }

    public void setDeltaWeight(double deltaWeight) {
        this.prevDeltaWeight = this.deltaWeight;
        this.deltaWeight = deltaWeight;
    }

    public double getPrevDeltaWeight() {
        return prevDeltaWeight;
    }
}