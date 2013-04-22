package org.wicht.api;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
    private final List<InputNeuron> input = new ArrayList<>();
    private final List<AbstractNeuron> output = new ArrayList<>();
    private final List<AbstractNeuron> hidden = new ArrayList<>();

    private NormalNeuron bias = new NormalNeuron();

    final double epsilon = 0.00000000001;
    final double learningRate = 0.9f;
    final double momentum = 0.7f;

    private NeuronFunction function;

    public void setFunction(NeuronFunction function) {
        this.function = function;
    }

    public void build(int i, int h, int o){
        //Build the intput layer
        for(int n = 0; n < i; ++n){
            input.add(new InputNeuron());
        }

        //Build the hidden layer
        for(int n = 0; n < h; ++n){
            NormalNeuron neuron = new NormalNeuron();
            neuron.setFunction(function);

            for(AbstractNeuron inputNeuron : input){
                Synapse synapse = new Synapse(inputNeuron, neuron);

                inputNeuron.addOutputSynapse(synapse);
                neuron.addInputSynapse(synapse);
            }

            neuron.setBiasConnection(new Synapse(bias, neuron));

            hidden.add(neuron);
        }

        //Build the output layer
        for(int n = 0; n < o; ++n){
            NormalNeuron neuron = new NormalNeuron();
            neuron.setFunction(function);

            for(AbstractNeuron inputNeuron : hidden){
                Synapse synapse = new Synapse(inputNeuron, neuron);

                inputNeuron.addOutputSynapse(synapse);
                neuron.addInputSynapse(synapse);
            }

            neuron.setBiasConnection(new Synapse(bias, neuron));

            output.add(neuron);
        }
    }

    public void train(List<List<Double>> inputs, List<List<Double>> expected, int maxIterations, double maxError){
        System.out.println("Start training");

        int iteration;

        double error = 1;
        for(iteration = 0; iteration < maxIterations && error > maxError; ++iteration){
            error = 0;

            for(int p = 0; p < inputs.size(); ++p){
                List<Double> output = activate(inputs.get(p));

                for(int j = 0; j < expected.get(p).size(); ++j){
                    error += Math.pow(output.get(j) - expected.get(p).get(j), 2);
                }

                backPropagate(expected.get(p));
            }
        }

        if(iteration == maxIterations){
            System.out.println("Failed to reach precision");
        }

        System.out.println("Sum of squared errors = " + error);
        System.out.println("##### EPOCH " + iteration);
        System.out.println();
    }

    private void backPropagate(List<Double> expected) {
        for(int i = 0; i < expected.size(); ++i){
            double d = expected.get(i);

            if(d < 0){
                expected.set(i, 0 + epsilon);
            } else if(d > 1){
                expected.set(i, 1 - epsilon);
            }
        }

        for(int i = 0; i < expected.size(); ++i){
            Neuron neuron = output.get(i);

            for(Synapse synapse : neuron.getInConnections()){
                double ak = neuron.getOutput();
                double ai = synapse.getInputNeuron().getOutput();
                double desired = expected.get(i);

                double partialDerivative = -ak * (1 -ak) * ai * (desired - ak);
                double delta = -learningRate * partialDerivative;
                double newWeight = synapse.getWeight() + delta;

                synapse.setDeltaWeight(delta);
                synapse.setWeight(newWeight + momentum * synapse.getPrevDeltaWeight());
            }
        }

        for(Neuron neuron : hidden){
            for(Synapse inSynapse : neuron.getInConnections()){
                double aj = neuron.getOutput();
                double ai = inSynapse.getInputNeuron().getOutput();

                double sumKoutputs = 0;
                int j = 0;

                for(Synapse outSynapse : neuron.getOutConnections()){
                    Neuron out_neu = outSynapse.getOutputNeuron();

                    double wjk = outSynapse.getWeight();
                    double desired = expected.get(j++);
                    double ak = out_neu.getOutput();
                    sumKoutputs += -(desired -ak) * ak * (1 - ak) * wjk;
                }

                double partialDerivative = aj * (1 - aj) * ai * sumKoutputs;
                double deltaWeight = -learningRate * partialDerivative;
                double newWeight = inSynapse.getWeight() + deltaWeight;

                inSynapse.setDeltaWeight(deltaWeight);
                inSynapse.setWeight(newWeight + momentum * inSynapse.getPrevDeltaWeight());
            }
        }
    }

    public List<Double> activate(List<Double> inputs){
        for(int i = 0; i < inputs.size(); ++i){
            input.get(i).setInput(inputs.get(i));
        }

        for(Neuron neuron : output){
            neuron.activate();
        }

        List<Double> results = new ArrayList<>();

        for(Neuron neuron : output){
           results.add(neuron.getOutput());
        }

        return results;
    }
}