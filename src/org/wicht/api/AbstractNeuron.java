package org.wicht.api;

import java.util.ArrayList;
import java.util.List;

public abstract class AbstractNeuron implements Neuron {
    private final List<Synapse> in = new ArrayList<>();
    private final List<Synapse> out = new ArrayList<>();

    private double output;

    public void addInputSynapse(Synapse synapse) {
        in.add(synapse);
    }

    public void addOutputSynapse(Synapse synapse) {
        out.add(synapse);
    }

    public List<Synapse> getInConnections() {
        return in;
    }

    public List<Synapse> getOutConnections() {
        return out;
    }

    @Override
    public double getOutput() {
        return output;
    }

    public void setOutput(double output) {
        this.output = output;
    }
}
