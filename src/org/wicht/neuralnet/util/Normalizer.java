package org.wicht.neuralnet.util;

import java.util.List;

public interface Normalizer {
    double normalize(double x);

    double denormalize(double x);

    List<Double> normalize(List<Double> inputs);

    List<List<Double>> normalize2D(List<List<Double>> inputs);

    List<Double> denormalize(List<Double> inputs);
}