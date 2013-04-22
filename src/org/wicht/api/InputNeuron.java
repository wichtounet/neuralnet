package org.wicht.api;

public class InputNeuron extends AbstractNeuron {
    double input;

    public void setInput(double input){
        this.input = input;
    }

    @Override
    public void activate() {
        //Nothing to activate in the input layer
    }

    @Override
    public double getOutput() {
        return input;
    }
}