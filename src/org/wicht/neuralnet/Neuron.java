package org.wicht.neuralnet;

import java.util.List;

public interface Neuron {
    void activate();

    double getOutput();

    List<Synapse> getInConnections();

    List<Synapse> getOutConnections();
}