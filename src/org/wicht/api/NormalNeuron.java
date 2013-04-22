package org.wicht.api;

public class NormalNeuron extends AbstractNeuron implements Neuron {
    private Synapse biasConnection;
    private double bias = -1;

    private double output;

    private NeuronFunction function;

    @Override
    public void activate(){
        double s = 0.0;

        for(Synapse synapse : in){
            Neuron inNeuron = synapse.getInputNeuron();

            inNeuron.activate();

            double w = synapse.getWeight();
            s += w * inNeuron.getOutput();
        }

        s += biasConnection.getWeight() * bias;

        output = function.activate(s);
    }

    public void setBiasConnection(Synapse biasConnection) {
        this.biasConnection = biasConnection;
    }

    @Override
    public double getOutput() {
        return output;
    }

    public void setFunction(NeuronFunction function) {
        this.function = function;
    }
}