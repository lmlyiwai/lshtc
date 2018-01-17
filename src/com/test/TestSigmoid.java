package com.test;

import static org.junit.Assert.*;

import org.junit.Test;

import com.tools.Sigmoid;

public class TestSigmoid {

	@Test
	public void test() {
		double[] x = new double[20];
		for(int i = 0; i < x.length; i++) {
			double tx = -3 + i * 0.3;
			x[i] = Sigmoid.sigmoid(tx, 1.0);
			System.out.println(tx + " --> " + x[i]);
			
			x[i] = Sigmoid.tanhx(tx, 1.0);
			System.out.println(tx + " --> " + x[i]);
			
			System.out.println();
		}
	}

}
