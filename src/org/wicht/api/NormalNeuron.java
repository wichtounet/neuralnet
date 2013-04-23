package org.wicht.api;

public class NormalNeuron extends AbstractNeuron {
    private NeuronFunction function;

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

    public void setFunction(NeuronFunction function) {
        this.function = function;
    }
}