package org.wicht.neuralnet.functions;

public class Identity implements NeuronFunction {
    @Override
    public double activate(double in) {
        return in;
    }
}