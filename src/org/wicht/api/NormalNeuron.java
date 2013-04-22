package org.wicht.api;

public class NormalNeuron extends AbstractNeuron implements Neuron {
    private Synapse biasConnection;
    private double bias = -1;

    private NeuronFunction function;

    @Override
    public void activate(){
        double s = 0.0;

        for(Synapse synapse : getInConnections()){
            Neuron inNeuron = synapse.getInputNeuron();

            double w = synapse.getWeight();
            s += w * inNeuron.getOutput();
        }

        s += biasConnection.getWeight() * bias;

        setOutput(function.activate(s));
    }

    public void setBiasConnection(Synapse biasConnection) {
        this.biasConnection = biasConnection;
    }

    public void setFunction(NeuronFunction function) {
        this.function = function;
    }
}