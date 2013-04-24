package org.wicht.neuralnet.util;

/**
 * Created with IntelliJ IDEA.
 * User: wichtounet
 * Date: 4/24/13
 * Time: 11:39 AM
 * To change this template use File | Settings | File Templates.
 */
public interface Normalizer {
    double normalize(double x);

    double denormalize(double x);
}
