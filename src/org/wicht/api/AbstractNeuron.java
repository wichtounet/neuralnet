package org.wicht.api;

import java.util.ArrayList;
import java.util.List;

public abstract class AbstractNeuron implements Neuron {
    protected List<Synapse> in = new ArrayList<>();
    private List<Synapse> out = new ArrayList<>();

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
}
