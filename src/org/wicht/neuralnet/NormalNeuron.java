package org.wicht.neuralnet;

import org.wicht.neuralnet.functions.ActivationFunction;

public class NormalNeuron extends AbstractNeuron {
    private ActivationFunction function;

    @Override
    public void activate() {
        double s = 0.0;

        for (Synapse synapse : getInConnections()) {
            Neuron inNeuron = synapse.getInputNeuron();

            double w = synapse.getWeight();
            s += w * inNeuron.getOutput();
        }

        setOutput(function.activate(s));
    }

    public void setFunction(ActivationFunction function) {
        this.function = function;
    }
}