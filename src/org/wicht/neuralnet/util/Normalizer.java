package org.wicht.neuralnet.util;

public class Normalizer {
    private final double LOW;
    private final double HIGH;

    private final double low;
    private final double high;

    public Normalizer(double low1, double high1, double low, double high) {
        LOW = low1;
        HIGH = high1;

        this.low = low;
        this.high = high;
    }

    public double normalize(double x) {
        return (x - low) / (high - low) * (HIGH - LOW) + LOW;
    }

    public double denormalize(double x) {
        return ((low - high) * x - HIGH * low + high * LOW) / (LOW - HIGH);
    }
}